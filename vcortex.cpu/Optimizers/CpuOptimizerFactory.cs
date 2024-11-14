using vcortex.Network;
using vcortex.Optimizers;

namespace vcortex.cpu.Optimizers;

public static class CpuOptimizerFactory
{
    public static IOptimizer Create(OptimizerConfig optimizer,  NetworkAcceleratorBuffers buffers, NetworkData networkData)
    {
        switch (optimizer)
        {
            case AdaDelta config:
                return new AdadeltaOptimizer(config, buffers, networkData);

            case AdaGrad config:
                return new AdagradOptimizer(config, buffers, networkData);

            case Adam config:
                return new AdamOptimizer(config, buffers, networkData);

            case RmsProp config:
                return new RMSpropOptimizer(config, buffers, networkData);

            case Sgd config:
                return new SgdOptimizer(config, buffers, networkData);

            case SgdMomentum config:
                return new SGDWithMomentumOptimizer(config, buffers, networkData);

            default:
                throw new ArgumentException($"Unsupported optimizer type: {optimizer.GetType().Name}");
        }
    }
}