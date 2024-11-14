namespace vcortex.Training;

public class StepDecayScheduler : ILearningRateScheduler
{
    private readonly StepDecay _stepDecay;

    public StepDecayScheduler(StepDecay stepDecay)
    {
        _stepDecay = stepDecay;
    }

    public float GetLearningRate(int epoch)
    {
        var stepCount = epoch / _stepDecay.StepSize;
        return _stepDecay.InitialLearningRate * MathF.Pow(_stepDecay.DecayFactor, stepCount);
    }
}