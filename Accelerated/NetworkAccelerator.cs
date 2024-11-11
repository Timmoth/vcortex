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

    public NetworkAccelerator(Network network)
    {
        Network = network;
        context = Context.Create(b =>
        {
            b.Default().EnableAlgorithms();
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
    }

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

    public float Train(List<(float[] inputs, float[] expectedOutputs)> batch, float learningRate = 0.02f)
    {
        for (int i = 0; i < batch.Count; i++)
        {
            var inputs = batch[i].inputs;
            Buffers.Activations.View.SubView(Network.NetworkData.ActivationCount * i, inputs.Length).CopyFromCPU(inputs);
        }

        foreach (var layer in Network._layers)
        {
            layer.Forward(this);
        }

        var finalLayer = Network._layers[^1];
        var actualOutputs = new float[finalLayer.NumOutputs];
        var errors = new float[finalLayer.NumOutputs];

        float sampleError = 0;

        for (int i = 0; i < batch.Count; i++)
        {
            var expectedOutputs = batch[i].expectedOutputs;

            Buffers.Activations.View.SubView(Network.NetworkData.ActivationCount * i + finalLayer.ActivationOutputOffset, finalLayer.NumOutputs).CopyToCPU(actualOutputs);

            for (var j = 0; j < actualOutputs.Length; j++)
            {
                var e = errors[j] = (actualOutputs[j] - expectedOutputs[j]);
                sampleError += e * e; // Sum of squared errors
            }

            Buffers.Errors.View.SubView(Network.NetworkData.ErrorCount * i + finalLayer.NextLayerErrorOffset, finalLayer.NumOutputs).CopyFromCPU(errors);
        }

        // Backward Pass
        for (var i = Network._layers.Length - 1; i >= 0; i--)
        {
            Network._layers[i].Backward(this);
        }

        foreach (var layer in Network._layers)
        {
            layer.AccumulateGradients(this);
        }

        return sampleError / finalLayer.NumOutputs;

    }
}