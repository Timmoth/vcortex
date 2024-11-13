using ILGPU;
using ILGPU.Runtime;
using vcortex.Core;

namespace vcortex.gpu;

public class NetworkAcceleratorBuffers : IDisposable
{
    public readonly MemoryBuffer1D<float, Stride1D.Dense> Activations;
    public readonly MemoryBuffer1D<float, Stride1D.Dense> Errors;
    public readonly MemoryBuffer1D<float, Stride1D.Dense> Gradients;
    public readonly MemoryBuffer1D<float, Stride1D.Dense> Inputs;
    public readonly MemoryBuffer1D<float, Stride1D.Dense> Outputs;
    public readonly MemoryBuffer1D<float, Stride1D.Dense> Parameters;
    public readonly int BatchSize;
    public NetworkAcceleratorBuffers(Accelerator accelerator, Network network, int batchSize)
    {
        BatchSize = batchSize;
        Parameters = accelerator.Allocate1D<float>(network.ParameterCount);
        Activations = accelerator.Allocate1D<float>(network.ActivationCount * batchSize);
        Gradients = accelerator.Allocate1D<float>(network.ParameterCount * batchSize);
        Errors = accelerator.Allocate1D<float>(network.ActivationCount * batchSize);
        Inputs = accelerator.Allocate1D<float>(network._layers[0].NumInputs * batchSize);
        Outputs = accelerator.Allocate1D<float>(network._layers[^1].NumOutputs * batchSize);
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