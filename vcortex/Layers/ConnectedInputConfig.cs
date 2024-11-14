using System.Text.Json.Serialization;
using vcortex.Input;

namespace vcortex.Layers;

public class ConnectedInputConfig : IInputConfig
{
    [JsonPropertyName("inputs")]
    public int NumInputs { get; set; }
}