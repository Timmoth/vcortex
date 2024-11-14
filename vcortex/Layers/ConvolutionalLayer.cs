using System.Text.Json.Serialization;

namespace vcortex.Layers;

public abstract class ConvolutionalLayer : Layer
{
    [JsonIgnore]
    internal int InputWidth { get; set; }
    [JsonIgnore]
    internal int InputHeight { get; set; }
    [JsonIgnore]
    internal int InputChannels { get; set; }
    [JsonIgnore]
    internal int OutputChannels { get; set; }

    [JsonIgnore]
    internal int OutputWidth { get; set; }
    [JsonIgnore]
    internal int OutputHeight { get; set; }
}