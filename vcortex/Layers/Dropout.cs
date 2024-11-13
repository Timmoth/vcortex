using vcortex.Input;

namespace vcortex.Layers;

public class Dropout : ConnectedLayer
{
    public float DropoutRate { get; set; }

    internal override void Connect(Layer prevLayer)
    {
        NumInputs = NumOutputs = prevLayer.NumOutputs;

        ParameterCount = 0;
        ParameterOffset = prevLayer.ParameterOffset + prevLayer.ParameterCount;

        ActivationInputOffset = prevLayer.ActivationOutputOffset;
        ActivationOutputOffset = prevLayer.ActivationOutputOffset + prevLayer.NumOutputs;
        CurrentLayerErrorOffset = prevLayer.NextLayerErrorOffset;
        NextLayerErrorOffset = prevLayer.CurrentLayerErrorOffset + NumInputs;
    }

    internal override void Connect(IInputConfig config)
    {
        if (config is not ConnectedInputConfig c) throw new Exception();

        NumInputs = NumOutputs = c.NumInputs;

        ParameterCount = 0;
        ParameterOffset = 0;

        ActivationInputOffset = 0;
        ActivationOutputOffset = c.NumInputs;
        CurrentLayerErrorOffset = 0;
        NextLayerErrorOffset = c.NumInputs;
    }
}