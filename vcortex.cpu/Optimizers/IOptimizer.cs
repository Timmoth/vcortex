using vcortex.Network;

namespace vcortex.cpu.Optimizers;

public interface IOptimizer
{
    public void Optimize(NetworkData networkData, NetworkAcceleratorBuffers buffers, float learningRate);

    public virtual void Reset()
    {
    }
}