using vcortex.Network;
using vcortex.Optimizers;

namespace vcortex.cpu.Optimizers;

public class SgdOptimizer : IOptimizer
{
    private readonly Sgd _sgd;

    public SgdOptimizer(Sgd sgd)
    {
        _sgd = sgd;
    }

    public void Optimize(NetworkData networkData, NetworkAcceleratorBuffers buffers, float learningRate)
    {

    }

    public void Dispose()
    {
    }
}