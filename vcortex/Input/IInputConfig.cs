using System.Text.Json.Serialization;
using vcortex.Layers;

namespace vcortex.Input;

[JsonPolymorphic()]
[JsonDerivedType(typeof(ConnectedInputConfig), "connected")]
[JsonDerivedType(typeof(ConvolutionInputConfig), "convolution")]
public interface IInputConfig
{
}