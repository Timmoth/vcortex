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