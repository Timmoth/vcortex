using vcortex.Layers;

namespace vcortex.cpu.Layers;

public class DenseLayer : IConnectedLayer
{
    private readonly Dense _dense;

    public DenseLayer(Dense dense)
    {
        _dense = dense;
    }

    public Layer Config => _dense;


    #region Kernel

    public virtual void FillRandom(INetworkAgent agent)
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