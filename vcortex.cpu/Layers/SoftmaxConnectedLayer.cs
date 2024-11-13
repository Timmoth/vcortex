using vcortex.Layers;

namespace vcortex.cpu.Layers;

public class SoftmaxConnectedLayer : IConnectedLayer
{
    private readonly Softmax _softmax;

    public SoftmaxConnectedLayer(Softmax softmax)
    {
        _softmax = softmax;
    }

    public Layer Config => _softmax;

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