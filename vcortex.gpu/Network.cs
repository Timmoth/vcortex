using vcortex.Core;
using vcortex.gpu.Layers;

namespace vcortex.gpu;

public class Network
{
    public readonly ILayer[] _layers;

    public NetworkData NetworkData;

    public Network(ILayer[] layers)
    {
        _layers = layers;
        NetworkData = new NetworkData(ActivationCount, ParameterCount);
    }

    public int ActivationCount => _layers.Sum(l => l.NumOutputs) + _layers[0].NumInputs;
    public int ParameterCount => _layers.Sum(l => l.ParameterCount);
}