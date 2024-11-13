using vcortex.Layers;

namespace vcortex.cpu.Layers;

public class DropoutLayer : IConnectedLayer
{
    private readonly Dropout _dropout;
    public DropoutLayer(Dropout dropout)
    {
        _dropout = dropout;
    }

    public Layer Config => _dropout;

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