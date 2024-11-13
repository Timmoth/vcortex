using vcortex.Network;

namespace vcortex.gpu.Optimizers;

public interface IOptimizer : IDisposable
{
    public void Compile(NetworkTrainer trainer);
    public void Optimize(NetworkData networkData, NetworkAcceleratorBuffers buffers, float learningRate);

    public virtual void Reset()
    {
    }
}