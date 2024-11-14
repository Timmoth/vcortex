using System.Text.Json.Serialization;

namespace vcortex.LearningRate;

[JsonPolymorphic()]
[JsonDerivedType(typeof(StepDecay), "step_decay")]
[JsonDerivedType(typeof(ExponentialDecay), "exponential_decay")]
[JsonDerivedType(typeof(ConstantLearningRate), "constant")]
public abstract class LearningRateScheduler
{
}

public class StepDecay : LearningRateScheduler
{
    [JsonPropertyName("lr")]
    public float InitialLearningRate { get; set; } = 0.01f;
    [JsonPropertyName("step")]
    public int StepSize { get; set; } = 10;
    [JsonPropertyName("decay")]
    public float DecayFactor { get; set; } = 0.5f;
}

public class ExponentialDecay : LearningRateScheduler
{
    [JsonPropertyName("lr")]
    public float InitialLearningRate { get; set; } = 0.01f;
    [JsonPropertyName("decay")]
    public float DecayRate { get; set; } = 0.05f;
}

public class ConstantLearningRate : LearningRateScheduler
{
    [JsonPropertyName("lr")]
    public float LearningRate { get; set; } = 0.01f;
}