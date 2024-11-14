namespace vcortex.Optimizers;

public interface IOptimizer : IDisposable
{
    public void Optimize(float learningRate);

    public virtual void Reset()
    {
    }
}