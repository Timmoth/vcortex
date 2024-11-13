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