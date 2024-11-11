using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;

namespace vcortex.Accelerated;

public class NetworkAccelerator : IDisposable
{
    private readonly Context context;
    private readonly Accelerator accelerator;

    public Network Network;
    public NetworkAcceleratorBuffers Buffers;
    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>> LoadInputsKernel { get; private set; }
    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, ArrayView<float>> LoadOutputsKernel { get; private set; }

    public NetworkAccelerator(Network network)
    {
        Network = network;
        context = Context.Create(b =>
        {
            b.Default().EnableAlgorithms().Math(MathMode.Fast);
        });

        bool useCuda = true;
        if (useCuda)
        {
            foreach (var device in context.GetCudaDevices())
            {
                Console.WriteLine(device.Name + " " + device.DeviceId);
            }

            accelerator = context.CreateCudaAccelerator(0);
        }
        else
        {
            context = Context.Create(b =>
            {
                b.Default().EnableAlgorithms().CPU();
            });

            accelerator = context.CreateCPUAccelerator(0);
        }

        Buffers = new NetworkAcceleratorBuffers(accelerator, network);

        foreach (var layer in network._layers)
        {
            layer.CompileKernels(accelerator);
        }
        
        LoadInputsKernel =
            accelerator.LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>>(
                LoadInputs);        
        
        LoadOutputsKernel =
            accelerator.LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                LoadOutputs);
        
        var inputLayer = Network._layers[0];
        var inputCount = inputLayer.LayerData.NumInputs * Network.NetworkData.BatchSize;
        flattenedInputs = new float[inputCount];
        
        var outputLayer = Network._layers[^1];
        var outputCount = outputLayer.LayerData.NumOutputs * Network.NetworkData.BatchSize;
        flattenedExpectedOutputs = new float[outputCount];
    }
    
    private readonly float[] flattenedInputs;
    private readonly float[] flattenedExpectedOutputs;

    public void Dispose()
    {
        accelerator.Dispose();
        context.Dispose();
        Buffers.Dispose();
    }

    public void InitWeights()
    {
        foreach (var networkLayer in Network._layers)
        {
            networkLayer.FillRandom(this);
        }
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
            for (int i = 0; i < batch.Count; i++)
            {
                Buffers.Activations.View.SubView(Network.NetworkData.ActivationCount * i, batch[i].Length).CopyFromCPU(batch[i]);
            }
            foreach (var layer in Network._layers)
            {
                layer.Forward(this);
            }

            for (int i = 0; i < batch.Count; i++)
            {
                var output = new float[finalLayer.NumOutputs];
                Buffers.Activations.View.SubView(Network.NetworkData.ActivationCount * i + finalLayer.ActivationOutputOffset, finalLayer.NumOutputs).CopyToCPU(output);
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

        activations[networkData.ActivationCount * batchIndex + inputIndex] = inputs[layerData.NumInputs * batchIndex + inputIndex];
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

            foreach (var layer in Network._layers)
            {
                layer.LayerData = layer.LayerData with
                {
                    Beta1  = 0.9f,
                    Beta2   = 0.999f,
                    Epsilon    = 1e-8f,
                    Timestep = layer.LayerData.Timestep + 1
                };
            }
            var inputLayer = Network._layers[0];
            var outputLayer = Network._layers[^1];
        
            for (int i = 0; i < batch.Count; i++)
            {
                Array.Copy(batch[i].inputs, 0, flattenedInputs, i * inputLayer.LayerData.NumInputs, inputLayer.LayerData.NumInputs);
            }
            
            Buffers.Inputs.View.CopyFromCPU(flattenedInputs);
            LoadInputsKernel(flattenedInputs.Length, Network.NetworkData, inputLayer.LayerData, Buffers.Inputs.View, Buffers.Activations.View);
            foreach (var layer in Network._layers)
            {
                layer.Forward(this);
            }
            var finalLayer = Network._layers[^1];
            
            for (int i = 0; i < batch.Count; i++)
            {
                Array.Copy(batch[i].expectedOutputs, 0, flattenedExpectedOutputs, i * outputLayer.LayerData.NumOutputs, outputLayer.LayerData.NumOutputs);
            }

            Buffers.Errors.View.MemSetToZero();
            Buffers.Outputs.View.CopyFromCPU(flattenedExpectedOutputs);
            LoadOutputsKernel(Network.NetworkData.BatchSize * outputLayer.LayerData.NumOutputs, Network.NetworkData, outputLayer.LayerData, Buffers.Outputs.View, Buffers.Activations.View, Buffers.Errors.View);
            Buffers.Outputs.View.CopyToCPU(flattenedExpectedOutputs);
            var sampleError = flattenedExpectedOutputs.Sum();
            
            // Backward Pass
            for (var i = Network._layers.Length - 1; i >= 0; i--)
            {
                Network._layers[i].Backward(this);
            }
            
            foreach (var layer in Network._layers)
            {
                layer.AccumulateGradients(this);
            }
            accelerator.Synchronize();

            //Console.WriteLine($" final sync: {stopwatch.ElapsedMilliseconds}ms");
            stopwatch.Restart();
            return sampleError / finalLayer.NumOutputs;
    }
}