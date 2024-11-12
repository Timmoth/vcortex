using vcortex.Accelerated;
using vcortex.Layers;

namespace vcortex;

public class Network
{
    public readonly ILayer[] _layers;

    public NetworkData NetworkData;

    public Network(ILayer[] layers, int batchSize)
    {
        _layers = layers;
        NetworkData = new NetworkData(0.01f, ActivationCount, ActivationCount, GradientCount, batchSize, 0.9f, 0.999f, 1e-8f,
            0);
    }

    public int ActivationCount => _layers.Sum(l => l.NumOutputs) + _layers[0].NumInputs;
    public int GradientCount => _layers.Sum(l => l.GradientCount);
    public int ParameterCount => _layers.Sum(l => l.ParameterCount);
}