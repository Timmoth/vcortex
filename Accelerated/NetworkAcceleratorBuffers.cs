using ILGPU;
using ILGPU.Runtime;

namespace vcortex.Accelerated;

public class NetworkAcceleratorBuffers : IDisposable
{
    public MemoryBuffer1D<float, Stride1D.Dense> Parameters;
    public MemoryBuffer1D<float, Stride1D.Dense> Activations;
    public MemoryBuffer1D<float, Stride1D.Dense> Gradients;
    public MemoryBuffer1D<float, Stride1D.Dense> Errors;

    public NetworkAcceleratorBuffers(Accelerator accelerator, Network network)
    {
        var batchSize = network.NetworkData.BatchSize;
        Parameters = accelerator.Allocate1D<float>(network.ParameterCount);
        Activations = accelerator.Allocate1D<float>(network.ActivationCount * batchSize);
        Gradients = accelerator.Allocate1D<float>(network.GradientCount * batchSize);
        Errors = accelerator.Allocate1D<float>(network.ActivationCount * batchSize);
    }

    public void Dispose()
    {
        Parameters.Dispose();
        Activations.Dispose();
        Gradients.Dispose();
        Errors.Dispose();
    }
}