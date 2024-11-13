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

    public void Optimize(NetworkData networkData, NetworkAcceleratorBuffers buffers, float learningRate)
    {

    }
}