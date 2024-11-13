namespace vcortex.LearningRate;

public abstract class LearningRateScheduler
{
}

public class StepDecay : LearningRateScheduler
{
    public float InitialLearningRate { get; set; } = 0.01f;
    public int StepSize { get; set; } = 10;
    public float DecayFactor { get; set; } = 0.5f;
}

public class ExponentialDecay : LearningRateScheduler
{
    public float InitialLearningRate { get; set; } = 0.01f;
    public float DecayRate { get; set; } = 0.05f;
}

public class ConstantLearningRate : LearningRateScheduler
{
    public float LearningRate { get; set; } = 0.01f;
}