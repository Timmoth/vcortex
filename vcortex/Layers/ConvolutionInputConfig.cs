using vcortex.Input;

namespace vcortex.Layers;

public class ConvolutionInputConfig : IInputConfig
{
    public int Width { get; set; }
    public int Height { get; set; }
    public bool Grayscale { get; set; }
}