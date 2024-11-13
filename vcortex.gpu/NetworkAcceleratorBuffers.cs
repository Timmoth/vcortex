using ILGPU;
using ILGPU.Runtime;
using vcortex.Core.Layers;

namespace vcortex.gpu;

public class NetworkAcceleratorBuffers : IDisposable
{
    public readonly MemoryBuffer1D<float, Stride1D.Dense> Activations;
    public readonly int BatchSize;
    public readonly MemoryBuffer1D<float, Stride1D.Dense> Errors;
    public readonly MemoryBuffer1D<float, Stride1D.Dense> Gradients;
    public readonly MemoryBuffer1D<float, Stride1D.Dense> Inputs;
    public readonly MemoryBuffer1D<float, Stride1D.Dense> Outputs;
    public readonly MemoryBuffer1D<float, Stride1D.Dense> Parameters;

    public NetworkAcceleratorBuffers(Accelerator accelerator, NetworkConfig network, int batchSize)
    {
        BatchSize = batchSize;
        Parameters = accelerator.Allocate1D<float>(network.NetworkData.ParameterCount);
        Activations = accelerator.Allocate1D<float>(network.NetworkData.ActivationCount * batchSize);
        Gradients = accelerator.Allocate1D<float>(network.NetworkData.ParameterCount * batchSize);
        Errors = accelerator.Allocate1D<float>(network.NetworkData.ActivationCount * batchSize);
        Inputs = accelerator.Allocate1D<float>(network.Layers[0].NumInputs * batchSize);
        Outputs = accelerator.Allocate1D<float>(network.Layers[^1].NumOutputs * batchSize);
    }

    public void Dispose()
    {
        Parameters.Dispose();
        Activations.Dispose();
        Gradients.Dispose();
        Errors.Dispose();
        Inputs.Dispose();
    }
}