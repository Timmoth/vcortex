namespace vcortex.Training;

public class ConstantLearningRateScheduler : ILearningRateScheduler
{
    private readonly ConstantLearningRate _constantLearningRate;

    public ConstantLearningRateScheduler(ConstantLearningRate constantLearningRate)
    {
        _constantLearningRate = constantLearningRate;
    }

    public float GetLearningRate(int epoch)
    {
        return _constantLearningRate.LearningRate;
    }
}