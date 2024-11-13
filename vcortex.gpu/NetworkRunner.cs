using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using vcortex.Core;

namespace vcortex.gpu;

public class NetworkRunner : INetworkAgent
{
    private readonly float[] _flattenedExpectedOutputs;
    private readonly float[] _flattenedInputs;
    private readonly Accelerator accelerator;
    private readonly Context context;
    private readonly NetworkAcceleratorBuffers _buffers;
    private readonly Network _network;

    public Accelerator Accelerator => accelerator;
    public Network Network => _network;
    public NetworkAcceleratorBuffers Buffers => _buffers;
    public bool IsTraining => false;
    
    public NetworkRunner(Network network, int batchSize)
    {
        _network = network;
        context = Context.Create(b => { b.Default().EnableAlgorithms().Math(MathMode.Fast); });

        var useCuda = true;
        if (useCuda)
        {
            foreach (var device in context.GetCudaDevices()) Console.WriteLine(device.Name + " " + device.DeviceId);

            accelerator = context.CreateCudaAccelerator(0);
        }
        else
        {
            context = Context.Create(b => { b.Default().EnableAlgorithms().CPU(); });

            accelerator = context.CreateCPUAccelerator(0);
        }

        _buffers = new NetworkAcceleratorBuffers(accelerator, network, batchSize);

        foreach (var layer in network._layers) layer.CompileKernels(this);
        
        LoadInputsKernel =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>>(
                    LoadInputs);

        var inputLayer = Network._layers[0];
        var inputCount = inputLayer.LayerData.NumInputs * _buffers.BatchSize;
        _flattenedInputs = new float[inputCount];

        var outputLayer = Network._layers[^1];
        var outputCount = outputLayer.LayerData.NumOutputs * _buffers.BatchSize;
        _flattenedExpectedOutputs = new float[outputCount];
    }
   
    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>> LoadInputsKernel { get; }
    

    public void Dispose()
    {
        accelerator.Dispose();
        context.Dispose();
        Buffers.Dispose();
    }

    public List<float[]> Predict(List<float[]> batchs)
    {
        var outputs = new List<float[]>();
        var batchSize = _buffers.BatchSize;
        var finalLayer = Network._layers[^1];

        // Divide the data into mini-batches
        for (var batchStart = 0; batchStart < batchs.Count; batchStart += batchSize)
        {
            // Get the current batch
            var batch = batchs.Skip(batchStart).Take(batchSize).ToList();
            var inputLayer = Network._layers[0];
            for (var i = 0; i < batch.Count; i++)
                Array.Copy(batch[i], 0, _flattenedInputs, i * inputLayer.LayerData.NumInputs,
                    inputLayer.LayerData.NumInputs);
            
            Buffers.Inputs.View.CopyFromCPU(_flattenedInputs);
            LoadInputsKernel(_flattenedInputs.Length, Network.NetworkData, inputLayer.LayerData, Buffers.Inputs.View,
                Buffers.Activations.View);
            
            foreach (var layer in Network._layers)
            {
                layer.Forward(this);
                accelerator.Synchronize();
            }
            
            for (var i = 0; i < batch.Count; i++)
            {
                var output = new float[finalLayer.NumOutputs];
                Buffers.Activations.View
                    .SubView(Network.NetworkData.ActivationCount * i + finalLayer.ActivationOutputOffset,
                        finalLayer.NumOutputs).CopyToCPU(output);
                outputs.Add(output);
            }
        }

        return outputs;
    }


    public static void LoadInputs(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> inputs,
        ArrayView<float> activations)
    {
        // Number of samples in the batch
        var batchIndex = index / layerData.NumInputs;
        var inputIndex = index % layerData.NumInputs;

        activations[networkData.ActivationCount * batchIndex + inputIndex] =
            inputs[layerData.NumInputs * batchIndex + inputIndex];
    }
}