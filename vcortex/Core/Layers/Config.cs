namespace vcortex.Core.Layers;

public abstract class Layer
{
    
}

public abstract class ConnectedLayer : Layer
{
    
}

public abstract class ConvolutionalLayer : Layer
{

}

public class Dense : ConnectedLayer
{
    public ActivationType Activation { get; set; }
    public int Neurons { get; set; }
}

public class Dropout : ConnectedLayer
{
    public float DropoutRate { get; set; }
}

public class Softmax : ConnectedLayer
{
    public int Neurons { get; set; }
}

public class Convolution : ConvolutionalLayer
{
    public int Stride { get; set; }
    public int Padding { get; set; }
    public int KernelsPerChannel { get; set; }
    public int KernelSize { get; set; }
    public ActivationType Activation { get; set; }
}

public class Maxpool : ConvolutionalLayer
{
    public float PoolSize { get; set; }
}

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
        if (_currentLayer == null && _nnInputConfig == null)
        {
            throw new InvalidOperationException($"Expected initial layer to be '{nameof(ConnectedLayer)}'");
        }

        _currentLayer = layer;
        _layers.Add(layer);
        return this;
    }

    public NetworkBuilder Add(ConvolutionalLayer layer)
    {
        if (_currentLayer == null && _cnnInputConfig == null)
        {
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

    public NetworkConfig Build(int batchSize)
    {
        return new NetworkConfig(_layers.ToArray(), batchSize);
    }
}

public class NetworkConfig
{
    public readonly Layer[] Layers;
    public NetworkData NetworkData;

    public NetworkConfig(Layer[] layers, int batchSize)
    {
        Layers = layers;
        var activationCount = 0;//Layers.Sum(l => l.NumOutputs) + Layers[0].NumInputs;
        var parameterCount = 0;//Layers.Sum(l => l.ParameterCount);
        NetworkData = new NetworkData(activationCount, parameterCount);
    }
}