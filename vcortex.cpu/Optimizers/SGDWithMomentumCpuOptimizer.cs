using vcortex.gpu.Optimizers;
using vcortex.Network;
using vcortex.Optimizers;

namespace vcortex.cpu.Optimizers;

public class SGDWithMomentumOptimizer : IOptimizer
{
    private readonly SgdMomentum _sgdMomentum;

    public SGDWithMomentumOptimizer(SgdMomentum sgdMomentum)
    {
        _sgdMomentum = sgdMomentum;
    }

    public void Dispose()
    {
        throw new NotImplementedException();
    }

    public void Optimize(float learningRate)
    {
        throw new NotImplementedException();
    }
}