using vcortex.Core.Optimizers;

namespace vcortex.gpu.Optimizers;

public static class GpuOptimizerFactory
{
    public static IOptimizer Create(OptimizerConfig optimizer)
    {
        switch (optimizer)
        {
            case AdaDelta config:
                return new AdadeltaOptimizer(config);

            case AdaGrad config:
                return new AdagradOptimizer(config);

            case Adam config:
                return new AdamOptimizer(config);

            case RmsProp config:
                return new RMSpropOptimizer(config);

            case Sgd config:
                return new SgdOptimizer(config);

            case SgdMomentum config:
                return new SGDWithMomentumOptimizer(config);

            default:
                throw new ArgumentException($"Unsupported optimizer type: {optimizer.GetType().Name}");
        }
    }
}