using vcortex.gpu.Optimizers;
using vcortex.Network;
using vcortex.Optimizers;

namespace vcortex.cpu.Optimizers;

public class AdadeltaOptimizer : IOptimizer
{
    private readonly AdaDelta _adaDelta;

    public AdadeltaOptimizer(AdaDelta adaDelta)
    {
        _adaDelta = adaDelta;
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