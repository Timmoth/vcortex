using ILGPU;
using ILGPU.Runtime;

namespace vcortex.Accelerated;

public class NetworkAcceleratorBuffers : IDisposable
{
    public MemoryBuffer1D<float, Stride1D.Dense> Parameters;
    public MemoryBuffer1D<float, Stride1D.Dense> Activations;
    public MemoryBuffer1D<float, Stride1D.Dense> Gradients;
    public MemoryBuffer1D<float, Stride1D.Dense> Errors;
    public MemoryBuffer1D<float, Stride1D.Dense> Inputs;
    public MemoryBuffer1D<float, Stride1D.Dense> Outputs;
    public MemoryBuffer1D<float, Stride1D.Dense> FirstMoment;
    public MemoryBuffer1D<float, Stride1D.Dense> SecondMoment;

    public NetworkAcceleratorBuffers(Accelerator accelerator, Network network)
    {
        var batchSize = network.NetworkData.BatchSize;
        Parameters = accelerator.Allocate1D<float>(network.ParameterCount);
        Activations = accelerator.Allocate1D<float>(network.ActivationCount * batchSize);
        Gradients = accelerator.Allocate1D<float>(network.GradientCount * batchSize);
        FirstMoment = accelerator.Allocate1D<float>(network.GradientCount);
        SecondMoment = accelerator.Allocate1D<float>(network.GradientCount);
        Errors = accelerator.Allocate1D<float>(network.ActivationCount * batchSize);
        Inputs = accelerator.Allocate1D<float>(network._layers[0].NumInputs * batchSize);
        Outputs = accelerator.Allocate1D<float>(network._layers[^1].NumOutputs * batchSize);
    }

    public void Dispose()
    {
        Parameters.Dispose();
        Activations.Dispose();
        Gradients.Dispose();
        FirstMoment.Dispose();
        SecondMoment.Dispose();
        Errors.Dispose();
        Inputs.Dispose();
    }
}