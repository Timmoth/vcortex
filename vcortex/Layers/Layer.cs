using System.Text.Json.Serialization;
using vcortex.Input;

namespace vcortex.Layers;

[JsonPolymorphic()]
[JsonDerivedType(typeof(Convolution), "convolution")]
[JsonDerivedType(typeof(Dense), "dense")]
[JsonDerivedType(typeof(Dropout), "dropout")]
[JsonDerivedType(typeof(Maxpool), "maxpool")]
[JsonDerivedType(typeof(Softmax), "softmax")]
public abstract class Layer
{
    [JsonIgnore]
    internal int NumInputs { get; set; }
    [JsonIgnore]
    internal int NumOutputs { get; set; }
    [JsonIgnore]
    internal int ActivationInputOffset { get; set; }
    [JsonIgnore]
    internal int ActivationOutputOffset { get; set; }
    [JsonIgnore]
    internal int CurrentLayerErrorOffset { get; set; }
    [JsonIgnore]
    internal int NextLayerErrorOffset { get; set; }
    [JsonIgnore]
    internal int ParameterCount { get; set; }
    [JsonIgnore]
    internal int ParameterOffset { get; set; }

    internal abstract void Connect(IInputConfig config);

    internal abstract void Connect(Layer prevLayer);
}