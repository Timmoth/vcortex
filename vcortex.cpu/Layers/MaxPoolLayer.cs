using vcortex.Layers;

namespace vcortex.cpu.Layers;

public class MaxPoolLayer : IConvolutionalLayer
{
    private readonly Maxpool _maxpool;

    public MaxPoolLayer(Maxpool maxpool)
    {
        _maxpool = maxpool;
    }

    public Layer Config => _maxpool;
    
    #region Kernels

    public void FillRandom(INetworkAgent agent)
    {
    }

    public void Forward(INetworkAgent agent)
    {

    }

    public void Backward(NetworkTrainer trainer)
    {

    }

    #endregion
}