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