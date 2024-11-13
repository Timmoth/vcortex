using vcortex.Input;

namespace vcortex.Layers;

public class Maxpool : ConvolutionalLayer
{
    public int PoolSize { get; set; }

    internal override void Connect(Layer prevLayer)
    {
        if (prevLayer is not ConvolutionalLayer l) throw new Exception();

        InputWidth = l.OutputWidth;
        InputHeight = l.OutputHeight;
        OutputChannels = InputChannels = l.OutputChannels;

        OutputHeight = InputHeight / PoolSize;
        OutputWidth = InputWidth / PoolSize;
        NumInputs = InputWidth * InputHeight * InputChannels;
        NumOutputs = OutputWidth * OutputHeight * OutputChannels;

        ParameterCount = 0;
        ParameterOffset = prevLayer.ParameterOffset + prevLayer.ParameterCount;

        ActivationInputOffset = prevLayer.ActivationOutputOffset;
        ActivationOutputOffset = prevLayer.ActivationOutputOffset + prevLayer.NumOutputs;
        CurrentLayerErrorOffset = prevLayer.NextLayerErrorOffset;
        NextLayerErrorOffset = prevLayer.CurrentLayerErrorOffset + NumInputs;
    }

    internal override void Connect(IInputConfig config)
    {
        if (config is not ConvolutionInputConfig c) throw new Exception();

        InputWidth = c.Width;
        InputHeight = c.Height;
        OutputChannels = InputChannels = c.Grayscale ? 1 : 3;

        OutputHeight = InputHeight / PoolSize;
        OutputWidth = InputWidth / PoolSize;
        NumInputs = InputWidth * InputHeight * InputChannels;
        NumOutputs = OutputWidth * OutputHeight * OutputChannels;

        ParameterCount = 0;
        ParameterOffset = 0;

        ActivationInputOffset = 0;
        ActivationOutputOffset = NumInputs;
        CurrentLayerErrorOffset = 0;
        NextLayerErrorOffset = NumInputs;
    }
}