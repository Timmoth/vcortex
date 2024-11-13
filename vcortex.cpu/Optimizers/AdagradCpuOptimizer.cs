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

  
    public void Optimize(NetworkData networkData, NetworkAcceleratorBuffers buffers, float learningRate)
    {

    }
}