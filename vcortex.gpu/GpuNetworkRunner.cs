using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using vcortex.gpu.Layers;
using vcortex.Layers;
using vcortex.Network;

namespace vcortex.gpu;

public enum GpuType
{
    OpenCl,
    Cuda
}

public class GpuNetworkRunner : IGpuNetworkAgent
{
    private readonly float[] _flattenedInputs;
    private readonly ILayer[] _layers;
    private readonly Context context;

    public GpuNetworkRunner(GpuType gpuType, int gpuIndex, NetworkConfig network, int batchSize)
    {
        Network = network;

        context = Context.Create(b => { b.Default().EnableAlgorithms().Math(MathMode.Fast); });

        if (gpuType == GpuType.Cuda)
        { 
            Accelerator = context.CreateCudaAccelerator(gpuIndex);
        }
        else
        {
            Accelerator = context.CreateCLAccelerator(gpuIndex);
        }

        Buffers = new NetworkAcceleratorBuffers(Accelerator, network, batchSize);

        _layers = network.Layers.Select(l => GpuLayerFactory.Create(Buffers, Accelerator, network.NetworkData, l)).ToArray();

        LoadInputsKernel =
            Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, ArrayView<float>, ArrayView<float>, int>(
                    LoadInputs);

        var inputLayer = _layers[0];
        var inputCount = inputLayer.Config.NumInputs * Buffers.BatchSize;
        _flattenedInputs = new float[inputCount];
    }

    public Action<Index1D, NetworkData, ArrayView<float>, ArrayView<float>, int> LoadInputsKernel { get; }

    public Accelerator Accelerator { get; }

    public NetworkConfig Network { get; }

    public NetworkAcceleratorBuffers Buffers { get; }

    public bool IsTraining => false;


    public void Dispose()
    {
        Accelerator.Dispose();
        context.Dispose();
        Buffers.Dispose();
    }

    public List<float[]> Predict(List<float[]> batchs)
    {
        var outputs = new List<float[]>();
        var batchSize = Buffers.BatchSize;
        var finalLayer = _layers[^1];

        // Divide the data into mini-batches
        for (var batchStart = 0; batchStart < batchs.Count; batchStart += batchSize)
        {
            // Get the current batch
            var batch = batchs.Skip(batchStart).Take(batchSize).ToList();
            var inputLayer = _layers[0];
            for (var i = 0; i < batch.Count; i++)
                Array.Copy(batch[i], 0, _flattenedInputs, i * inputLayer.Config.NumInputs,
                    inputLayer.Config.NumInputs);

            Buffers.Inputs.View.CopyFromCPU(_flattenedInputs);
            LoadInputsKernel(_flattenedInputs.Length, Network.NetworkData, Buffers.Inputs.View,
                Buffers.Activations.View, inputLayer.Config.NumInputs);

            foreach (var layer in _layers)
            {
                layer.Forward();
                Accelerator.Synchronize();
            }

            for (var i = 0; i < batch.Count; i++)
            {
                var output = new float[finalLayer.Config.NumOutputs];
                Buffers.Activations.View
                    .SubView(Network.NetworkData.ActivationCount * i + finalLayer.Config.ActivationOutputOffset,
                        finalLayer.Config.NumOutputs).CopyToCPU(output);
                outputs.Add(output);
            }
        }

        return outputs;
    }


    public static void LoadInputs(
        Index1D index,
        NetworkData networkData,
        ArrayView<float> inputs,
        ArrayView<float> activations, int numInputs)
    {
        // Number of samples in the batch
        var batchIndex = index / numInputs;
        var inputIndex = index % numInputs;

        activations[networkData.ActivationCount * batchIndex + inputIndex] =
            inputs[numInputs * batchIndex + inputIndex];
    }
    
    #region Io

    public void SaveParametersToDisk(string filePath)
    {
        using var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write);
        using var writer = new BinaryWriter(stream);
        // Write the number of arrays to allow easy deserialization
        writer.Write(Network.NetworkData.ParameterCount);
        var parameters = new float[Network.NetworkData.ParameterCount];
        Buffers.Parameters.View.CopyToCPU(parameters);
        foreach (var value in parameters) writer.Write(value);
        
    }

    public void ReadParametersFromDisk(string filePath)
    {
        using var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
        using var reader = new BinaryReader(stream);
        // Read the number of arrays
        var length = reader.ReadInt32();
        var parameters = new float[length];

        for (var j = 0; j < length; j++) parameters[j] = reader.ReadSingle();
        Buffers.Parameters.View.CopyFromCPU(parameters);
    }

    public float[] GetParameters()
    {
        var parameters = new float[Network.NetworkData.ParameterCount];
        Buffers.Parameters.View.CopyToCPU(parameters);
        return parameters;
    }

    public void LoadParameters(float[] parameters)
    {
        Buffers.Parameters.View.CopyFromCPU(parameters);
    }

    #endregion
}