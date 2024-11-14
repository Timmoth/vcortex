namespace vcortex;

public interface ILossFunction : IDisposable
{
    public float Apply(List<(float[] inputs, float[] expectedOutputs)> batch);
}