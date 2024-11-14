using vcortex.Input;
using vcortex.Layers;

namespace vcortex.Network;

public class NetworkBuilder
{
    private readonly ConvolutionInputConfig? _cnnInputConfig;
    private readonly List<Layer> _layers = new();
    private readonly ConnectedInputConfig? _nnInputConfig;
    private Layer? _currentLayer;

    public NetworkBuilder(IInputConfig config)
    {
        if (config is ConvolutionInputConfig convolutionInputConfig)
            _cnnInputConfig = convolutionInputConfig;
        else if (config is ConnectedInputConfig nnInputConfig)
            _nnInputConfig = nnInputConfig;
        else
            throw new InvalidOperationException("Invalid input config");
    }

    public NetworkBuilder(ConvolutionInputConfig config)
    {
        _cnnInputConfig = config;
    }

    public NetworkBuilder Add(ConnectedLayer layer)
    {
        if (_currentLayer == null)
            if (_nnInputConfig == null)
                throw new InvalidOperationException($"Expected initial layer to be '{nameof(ConnectedLayer)}'");

        _currentLayer = layer;
        _layers.Add(layer);
        return this;
    }

    public NetworkBuilder Add(ConvolutionalLayer layer)
    {
        if (_currentLayer == null)
        {
            if (_cnnInputConfig == null)
                throw new InvalidOperationException($"Expected initial layer to be '{nameof(ConvolutionalLayer)}'");
        }
        else if (_currentLayer is not ConvolutionalLayer)
        {
            throw new InvalidOperationException(
                $"Can't connect a '{nameof(ConvolutionalLayer)}' to a '{nameof(ConnectedLayer)}'");
        }

        _currentLayer = layer;
        _layers.Add(layer);
        return this;
    }

    public NetworkConfig Build()
    {
        if (_cnnInputConfig != null) return new NetworkConfig(_layers.ToArray(), _cnnInputConfig);

        if (_nnInputConfig != null) return new NetworkConfig(_layers.ToArray(), _nnInputConfig);

        throw new Exception();
    }
}