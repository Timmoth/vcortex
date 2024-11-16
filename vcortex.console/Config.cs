using System.Text.Json.Serialization;
using vcortex.Network;
using vcortex.Training;

namespace vcortex.console;

public class Config
{
    [JsonPropertyName("network")]
    public NetworkConfig Network { get; set; }
    [JsonPropertyName("training")]
    public TrainConfig Training { get; set; }

    [JsonPropertyName("train_file")]
    public string TrainingFile { get; set; }
    [JsonPropertyName("test_file")]
    public string TestingFile { get; set; }
    
    [JsonPropertyName("parameters_file")]
    public string ParametersFile { get; set; }
}