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