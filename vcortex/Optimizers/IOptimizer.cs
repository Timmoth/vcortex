namespace vcortex.gpu.Optimizers;

public interface IOptimizer : IDisposable
{
    public void Optimize(float learningRate);

    public virtual void Reset()
    {
    }
}