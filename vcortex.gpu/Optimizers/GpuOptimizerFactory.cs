using vcortex.Optimizers;

namespace vcortex.gpu.Optimizers;

public static class GpuOptimizerFactory
{
    public static IOptimizer Create(OptimizerConfig optimizer, GpuNetworkTrainer trainer)
    {
        switch (optimizer)
        {
            case AdaDelta config:
                return new AdadeltaOptimizer(config, trainer);

            case AdaGrad config:
                return new AdagradOptimizer(config, trainer);

            case Adam config:
                return new AdamOptimizer(config, trainer);

            case RmsProp config:
                return new RMSpropOptimizer(config, trainer);

            case Sgd config:
                return new SgdOptimizer(config, trainer);

            case SgdMomentum config:
                return new SGDWithMomentumOptimizer(config, trainer);

            default:
                throw new ArgumentException($"Unsupported optimizer type: {optimizer.GetType().Name}");
        }
    }
}