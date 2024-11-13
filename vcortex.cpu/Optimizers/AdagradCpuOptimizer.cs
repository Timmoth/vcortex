using vcortex.gpu.Optimizers;
using vcortex.Network;
using vcortex.Optimizers;

namespace vcortex.cpu.Optimizers;

public class AdagradOptimizer : IOptimizer
{
    private readonly AdaGrad _adaGrad;
    public AdagradOptimizer(AdaGrad adaGrad)
    {
        _adaGrad = adaGrad;
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