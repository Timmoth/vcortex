using vcortex.Input;

namespace vcortex.Layers;

public class Convolution : ConvolutionalLayer
{
    public int Stride { get; set; }
    public int Padding { get; set; }
    public int KernelsPerChannel { get; set; }
    public int KernelSize { get; set; }
    public ActivationType Activation { get; set; }

    internal override void Connect(IInputConfig config)
    {
        if (config is not ConvolutionInputConfig c) throw new Exception();

        InputChannels = c.Grayscale ? 1 : 3;
        InputWidth = c.Width;
        InputHeight = c.Height;
        NumInputs = InputWidth * InputHeight * InputChannels;
        OutputWidth = (InputWidth - KernelSize + 2 * Padding) / Stride + 1;
        OutputHeight = (InputHeight - KernelSize + 2 * Padding) / Stride + 1;
        OutputChannels = KernelsPerChannel * InputChannels;

        NumOutputs = OutputWidth * OutputHeight * OutputChannels;

        ParameterCount = OutputChannels * KernelSize * KernelSize;
        ParameterOffset = 0;

        ActivationInputOffset = 0;
        ActivationOutputOffset = NumInputs;
        CurrentLayerErrorOffset = 0;
        NextLayerErrorOffset = NumInputs;
    }

    internal override void Connect(Layer prevLayer)
    {
        if (prevLayer is not ConvolutionalLayer l) throw new Exception();

        InputChannels = l.OutputChannels;
        InputWidth = l.OutputWidth;
        InputHeight = l.OutputHeight;

        NumInputs = InputWidth * InputHeight * InputChannels;
        OutputWidth = (InputWidth - KernelSize + 2 * Padding) / Stride + 1;
        OutputHeight = (InputHeight - KernelSize + 2 * Padding) / Stride + 1;
        OutputChannels = KernelsPerChannel * InputChannels;
        NumOutputs = OutputWidth * OutputHeight * OutputChannels;

        ParameterCount = OutputChannels * KernelSize * KernelSize;
        ParameterOffset = l.ParameterOffset + l.ParameterCount;

        ActivationInputOffset = l.ActivationOutputOffset;
        ActivationOutputOffset = l.ActivationOutputOffset + l.NumOutputs;
        CurrentLayerErrorOffset = l.NextLayerErrorOffset;
        NextLayerErrorOffset = l.CurrentLayerErrorOffset + NumInputs;
    }
}