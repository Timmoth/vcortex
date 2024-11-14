namespace vcortex.Training;

public class ExponentialDecayScheduler : ILearningRateScheduler
{
    private readonly ExponentialDecay _exponentialDecay;

    public ExponentialDecayScheduler(ExponentialDecay exponentialDecay)
    {
        _exponentialDecay = exponentialDecay;
    }

    public float GetLearningRate(int epoch)
    {
        return _exponentialDecay.InitialLearningRate * MathF.Exp(-_exponentialDecay.DecayRate * epoch);
    }
}