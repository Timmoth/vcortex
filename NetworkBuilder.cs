using vcortex.Layers;
using vcortex.Layers.Connected;
using vcortex.Layers.Convolution;

namespace vcortex;

public class NetworkBuilder
{
    private readonly ConvolutionInputConfig? _cnnInputConfig;
    private readonly List<ILayer> _layers = new();
    private readonly ConnectedInputConfig? _nnInputConfig;
    private ILayer? _currentLayer;

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

    public NetworkBuilder Add(IConnectedLayer layer)
    {
        if (_currentLayer == null)
        {
            if (_nnInputConfig == null)
                throw new InvalidOperationException($"Expected initial layer to be '{nameof(IConvolutionalLayer)}'");

            layer.Connect(_nnInputConfig);
        }
        else
        {
            layer.Connect(_currentLayer);
        }

        _currentLayer = layer;
        _layers.Add(layer);
        layer.FillRandom();
        return this;
    }

    public NetworkBuilder Add(IConvolutionalLayer layer)
    {
        if (_currentLayer == null)
        {
            if (_cnnInputConfig == null)
                throw new InvalidOperationException($"Expected initial layer to be '{nameof(IConnectedLayer)}'");

            layer.Connect(_cnnInputConfig);
        }
        else if (_currentLayer is IConvolutionalLayer convolutionalLayer)
        {
            layer.Connect(convolutionalLayer);
        }
        else
        {
            throw new InvalidOperationException(
                $"Can't connect a '{nameof(IConvolutionalLayer)}' to a '{nameof(IConnectedLayer)}'");
        }

        _currentLayer = layer;
        _layers.Add(layer);
        layer.FillRandom();
        return this;
    }

    public Network Build()
    {
        return new Network(_layers.ToArray());
    }
}