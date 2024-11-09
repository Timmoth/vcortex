namespace vcortex.Layers.Convolution;

public interface IConvolutionalLayer : ILayer
{
    public int InputWidth { get; }
    public int InputHeight { get; }
    public int OutputWidth { get; }
    public int OutputHeight { get; }
    public int InputChannels { get; }
    public int OutputChannels { get; }

    public void Connect(IConvolutionalLayer prevLayer);
    public void Connect(ConvolutionInputConfig config);
}