using vcortex.Layers;

namespace vcortex.cpu.Layers;

public class KernelConvolutionLayer : IConvolutionalLayer
{
    private readonly Convolution _convolution;
    public KernelConvolutionLayer(Convolution convolution)
    {
        _convolution = convolution;
    }

    public Layer Config => _convolution;
    public void FillRandom()
    {
        throw new NotImplementedException();
    }

    public void Forward()
    {
        throw new NotImplementedException();
    }

    public void Backward()
    {
        throw new NotImplementedException();
    }

    public bool IsTraining { get; set; }
}