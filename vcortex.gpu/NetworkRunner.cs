using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using vcortex.gpu.Layers;
using vcortex.Network;

namespace vcortex.gpu;

public class NetworkRunner : INetworkAgent
{
    private readonly float[] _flattenedInputs;
    private readonly ILayer[] _layers;
    private readonly Context context;

    public NetworkRunner(NetworkConfig network, int batchSize)
    {
        Network = network;
        _layers = network.Layers.Select(GpuLayerFactory.Create).ToArray();

        context = Context.Create(b => { b.Default().EnableAlgorithms().Math(MathMode.Fast); });

        var useCuda = true;
        if (useCuda)
        {
            foreach (var device in context.GetCudaDevices()) Console.WriteLine(device.Name + " " + device.DeviceId);

            Accelerator = context.CreateCudaAccelerator(0);
        }
        else
        {
            context = Context.Create(b => { b.Default().EnableAlgorithms().CPU(); });

            Accelerator = context.CreateCPUAccelerator(0);
        }

        Buffers = new NetworkAcceleratorBuffers(Accelerator, network, batchSize);

        foreach (var layer in _layers) layer.CompileKernels(this);

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
                layer.Forward(this);
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
}