using System.Text.Json.Serialization;
using vcortex.Optimizers;

namespace vcortex.Training;

public class TrainConfig
{
    [JsonPropertyName("epochs")]
    public required int Epochs { get; set; }
    [JsonPropertyName("lr_schedule")]
    public required LearningRateScheduler Scheduler { get; set; }
    [JsonPropertyName("optimizer")]
    public required OptimizerConfig Optimizer { get; set; }
    [JsonPropertyName("loss")]
    public required LossFunction LossFunction { get; set; }
    [JsonPropertyName("batch")]
    public required int BatchSize { get; set; }
}