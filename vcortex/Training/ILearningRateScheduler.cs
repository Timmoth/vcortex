namespace vcortex.Training;

public interface ILearningRateScheduler
{
    public float GetLearningRate(int epoch);
}