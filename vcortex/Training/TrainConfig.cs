using System.Text.Json.Serialization;
using vcortex.Optimizers;

namespace vcortex.Training;

public enum Platform
{
    Cpu,
    Cuda,
    OpenCl
}

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
    
    [JsonPropertyName("platform")]
    public required Platform Platform { get; set; }
    
    [JsonPropertyName("gpu_index")]
    public required int? GpuIndex { get; set; }
}