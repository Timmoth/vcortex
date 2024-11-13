using vcortex.Input;
using vcortex.Layers;

namespace vcortex.Network;

public class NetworkConfig
{
    public readonly Layer[] Layers;
    public NetworkData NetworkData;

    public NetworkConfig(Layer[] layers, IInputConfig config)
    {
        Layers = layers;

        layers[0].Connect(config);
        for (var i = 1; i < Layers.Length; i++) layers[i].Connect(layers[i - 1]);

        var activationCount = layers.Sum(l => l.NumOutputs) + Layers[0].NumInputs;
        var parameterCount = layers.Sum(l => l.ParameterCount);
        NetworkData = new NetworkData(activationCount, parameterCount);
    }
}