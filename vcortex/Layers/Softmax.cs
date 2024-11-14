using System.Text.Json.Serialization;
using vcortex.Input;

namespace vcortex.Layers;

public class Softmax : ConnectedLayer
{
    [JsonPropertyName("neurons")]
    public int Neurons { get; set; }
    [JsonIgnore]
    internal int BiasOffset { get; set; }

    internal override void Connect(Layer prevLayer)
    {
        NumOutputs = Neurons;
        NumInputs = prevLayer.NumOutputs;

        ParameterCount = NumOutputs * NumInputs + NumOutputs;
        ParameterOffset = prevLayer.ParameterOffset + prevLayer.ParameterCount;
        BiasOffset = NumInputs * NumOutputs;

        ActivationInputOffset = prevLayer.ActivationOutputOffset;
        ActivationOutputOffset = prevLayer.ActivationOutputOffset + prevLayer.NumOutputs;
        CurrentLayerErrorOffset = prevLayer.NextLayerErrorOffset;
        NextLayerErrorOffset = prevLayer.CurrentLayerErrorOffset + NumInputs;
    }

    internal override void Connect(IInputConfig config)
    {
        if (config is not ConnectedInputConfig c) throw new Exception();

        NumOutputs = Neurons;
        NumInputs = c.NumInputs;
        BiasOffset = NumInputs * NumOutputs;

        ParameterCount = NumOutputs * NumInputs + NumOutputs;
        ParameterOffset = 0;

        ActivationInputOffset = 0;
        ActivationOutputOffset = c.NumInputs;
        CurrentLayerErrorOffset = 0;
        NextLayerErrorOffset = c.NumInputs;
    }
}