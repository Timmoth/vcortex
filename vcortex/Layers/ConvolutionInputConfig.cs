using System.Text.Json.Serialization;
using vcortex.Input;

namespace vcortex.Layers;

public class ConvolutionInputConfig : IInputConfig
{
    [JsonPropertyName("width")]
    public int Width { get; set; }
    [JsonPropertyName("height")]
    public int Height { get; set; }
    [JsonPropertyName("is_grayscale")]
    public bool Grayscale { get; set; }
}