using vcortex.Input;

namespace vcortex.Layers;

public class Dense : ConnectedLayer
{
    public ActivationType Activation { get; set; }
    public int Neurons { get; set; }

    public int BiasOffset { get; set; }

    internal override void Connect(Layer prevLayer)
    {
        NumInputs = prevLayer.NumOutputs;
        NumOutputs = Neurons;

        BiasOffset = NumInputs * NumOutputs;

        ParameterCount = NumOutputs * NumInputs + NumOutputs;
        ParameterOffset = prevLayer.ParameterOffset + prevLayer.ParameterCount;

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