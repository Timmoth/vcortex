namespace vcortex.LearningRate;

public static class LearningRateSchedulerFactory
{
    public static ILearningRateScheduler Create(LearningRateScheduler scheduler)
    {
        switch (scheduler)
        {
            case ExponentialDecay exponential:
                return new ExponentialDecayScheduler(exponential);

            case StepDecay step:
                return new StepDecayScheduler(step);

            case ConstantLearningRate constant:
                return new ConstantLearningRateScheduler(constant);

            default:
                throw new ArgumentException($"Unsupported scheduler type: {scheduler.GetType().Name}");
        }
    }
}