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

    public virtual void FillRandom(INetworkAgent agent)
    {

    }

    public void Forward(INetworkAgent agent)
    {

    }

    public void Backward(NetworkTrainer trainer)
    {

    }
}