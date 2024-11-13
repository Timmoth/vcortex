using vcortex.gpu.Optimizers;
using vcortex.Network;
using vcortex.Optimizers;

namespace vcortex.cpu.Optimizers;

public class AdamOptimizer : IOptimizer
{
    private readonly Adam _adam;
    public int Timestep;

    public AdamOptimizer(Adam adam)
    {
        _adam = adam;
    }
    public void Optimize(NetworkData networkData, NetworkAcceleratorBuffers buffers, float learningRate)
    {
        Timestep++;
    }

    public void Optimize(float learningRate)
    {
        throw new NotImplementedException();
    }

    public void Reset()
    {
        Timestep = 0;
    }

    public void Dispose()
    {
        throw new NotImplementedException();
    }
}