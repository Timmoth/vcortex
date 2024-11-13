namespace vcortex.LearningRate;

public interface ILearningRateScheduler
{
    public float GetLearningRate(int epoch);
}