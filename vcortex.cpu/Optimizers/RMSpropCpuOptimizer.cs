using vcortex.Network;
using vcortex.Optimizers;

namespace vcortex.cpu.Optimizers;

public class RMSpropOptimizer : IOptimizer
{
    private readonly RmsProp _rmsProp;
    public RMSpropOptimizer(RmsProp rmsProp)
    {
        _rmsProp = rmsProp;
    }
    public void Optimize(NetworkData networkData, NetworkAcceleratorBuffers buffers, float learningRate)
    {
    }
}