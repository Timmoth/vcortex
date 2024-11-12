using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;

namespace vcortex.Accelerated;

public class NetworkAccelerator : IDisposable
{
    private readonly float[] _flattenedExpectedOutputs;

    private readonly float[] _flattenedInputs;
    public readonly Accelerator accelerator;
    private readonly Context context;
    public NetworkAcceleratorBuffers Buffers;

    public Network Network;

    public NetworkAccelerator(Network network)
    {
        Network = network;
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

        Buffers = new NetworkAcceleratorBuffers(accelerator, network);

        foreach (var layer in network._layers) layer.CompileKernels(accelerator);

        LoadInputsKernel =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>>(
                    LoadInputs);

        LoadOutputsKernel =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>>(
                    LoadOutputs);

        var inputLayer = Network._layers[0];
        var inputCount = inputLayer.LayerData.NumInputs * Network.NetworkData.BatchSize;
        _flattenedInputs = new float[inputCount];

        var outputLayer = Network._layers[^1];
        var outputCount = outputLayer.LayerData.NumOutputs * Network.NetworkData.BatchSize;
        _flattenedExpectedOutputs = new float[outputCount];
    }

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>> LoadInputsKernel { get; }

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, ArrayView<float>>
        LoadOutputsKernel { get; }

    public void Dispose()
    {
        accelerator.Dispose();
        context.Dispose();
        Buffers.Dispose();
    }

    public void InitWeights()
    {
        foreach (var networkLayer in Network._layers) networkLayer.FillRandom(this);
    }

    public List<float[]> Predict(List<float[]> inputs)
    {
        var outputs = new List<float[]>();
        var batchSize = Network.NetworkData.BatchSize;
        var finalLayer = Network._layers[^1];

        // Divide the data into mini-batches
        for (var batchStart = 0; batchStart < inputs.Count; batchStart += batchSize)
        {
            // Get the current batch
            var batch = inputs.Skip(batchStart).Take(batchSize).ToList();
            for (var i = 0; i < batch.Count; i++)
                Buffers.Activations.View.SubView(Network.NetworkData.ActivationCount * i, batch[i].Length)
                    .CopyFromCPU(batch[i]);
            foreach (var layer in Network._layers) layer.Forward(this);

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

    public static void LoadOutputs(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> outputs,
        ArrayView<float> activations,
        ArrayView<float> errors)
    {
        // Number of samples in the batch
        var batchIndex = index / layerData.NumOutputs;
        var outputIndex = index % layerData.NumOutputs;
        var activationOutputOffset = batchIndex * networkData.ActivationCount + layerData.ActivationOutputOffset;
        var nextErrorOffset = batchIndex * networkData.ErrorCount + layerData.NextLayerErrorOffset;

        var expected = outputs[layerData.NumOutputs * batchIndex + outputIndex];
        var actual = activations[activationOutputOffset + outputIndex];
        var error = actual - expected;

        outputs[layerData.NumOutputs * batchIndex + outputIndex] = error * error;
        errors[nextErrorOffset + outputIndex] = error;
    }

    public float Train(List<(float[] inputs, float[] expectedOutputs)> batch)
    {
        var stopwatch = Stopwatch.StartNew();

        Network.NetworkData = new NetworkData(Network.NetworkData.LearningRate, Network.NetworkData.ActivationCount,
            Network.NetworkData.ActivationCount, Network.NetworkData.GradientCount, Network.NetworkData.BatchSize, 0.9f,
            0.999f, 1e-8f,
            Network.NetworkData.Timestep + 1);

        var inputLayer = Network._layers[0];
        var outputLayer = Network._layers[^1];

        for (var i = 0; i < batch.Count; i++)
            Array.Copy(batch[i].inputs, 0, _flattenedInputs, i * inputLayer.LayerData.NumInputs,
                inputLayer.LayerData.NumInputs);

        Buffers.Inputs.View.CopyFromCPU(_flattenedInputs);
        LoadInputsKernel(_flattenedInputs.Length, Network.NetworkData, inputLayer.LayerData, Buffers.Inputs.View,
            Buffers.Activations.View);
        foreach (var layer in Network._layers)
        {
            layer.Forward(this);
            accelerator.Synchronize();
        }

        var finalLayer = Network._layers[^1];

        for (var i = 0; i < batch.Count; i++)
            Array.Copy(batch[i].expectedOutputs, 0, _flattenedExpectedOutputs, i * outputLayer.LayerData.NumOutputs,
                outputLayer.LayerData.NumOutputs);

        Buffers.Errors.View.MemSetToZero();
        Buffers.Outputs.View.CopyFromCPU(_flattenedExpectedOutputs);
        LoadOutputsKernel(Network.NetworkData.BatchSize * outputLayer.LayerData.NumOutputs, Network.NetworkData,
            outputLayer.LayerData, Buffers.Outputs.View, Buffers.Activations.View, Buffers.Errors.View);
        Buffers.Outputs.View.CopyToCPU(_flattenedExpectedOutputs);
        var sampleError = _flattenedExpectedOutputs.Sum();

        // Backward Pass
        for (var i = Network._layers.Length - 1; i >= 0; i--)
        {
            Network._layers[i].Backward(this);
            accelerator.Synchronize();
        }

        foreach (var layer in Network._layers)
        {
            layer.AccumulateGradients(this);
            accelerator.Synchronize();
        }

        //Console.WriteLine($" final sync: {stopwatch.ElapsedMilliseconds}ms");
        stopwatch.Restart();
        return sampleError / finalLayer.NumOutputs;
    }
}