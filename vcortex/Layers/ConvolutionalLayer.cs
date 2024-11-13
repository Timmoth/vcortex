namespace vcortex.Layers;

public abstract class ConvolutionalLayer : Layer
{
    internal int InputWidth { get; set; }
    internal int InputHeight { get; set; }
    internal int InputChannels { get; set; }
    internal int OutputChannels { get; set; }

    internal int OutputWidth { get; set; }
    internal int OutputHeight { get; set; }
}