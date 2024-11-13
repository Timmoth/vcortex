namespace vcortex.Core.Layers;

public abstract class Layer
{
    internal int NumInputs { get; set; }
    internal int NumOutputs { get; set; }
    internal int ActivationInputOffset { get; set; }
    internal int ActivationOutputOffset { get; set; }
    internal int CurrentLayerErrorOffset { get; set; }
    internal int NextLayerErrorOffset { get; set; }
    internal int ParameterCount { get; set; }
    internal int ParameterOffset { get; set; }

    internal abstract void Connect(IInputConfig config);

    internal abstract void Connect(Layer prevLayer);
}

public abstract class ConnectedLayer : Layer
{
}

public abstract class ConvolutionalLayer : Layer
{
    internal int InputWidth { get; set; }
    internal int InputHeight { get; set; }
    internal int InputChannels { get; set; }
    internal int OutputChannels { get; set; }

    internal int OutputWidth { get; set; }
    internal int OutputHeight { get; set; }
}

public class Dense : ConnectedLayer
{
    public ActivationType Activation { get; set; }
    public int Neurons { get; set; }

    public int BiasOffset { get; set; }

    internal override void Connect(Layer prevLayer)
    {
        NumInputs = prevLayer.NumOutputs;
        NumOutputs = Neurons;

        BiasOffset = NumInputs * NumOutputs;

        ParameterCount = NumOutputs * NumInputs + NumOutputs;
        ParameterOffset = prevLayer.ParameterOffset + prevLayer.ParameterCount;

        ActivationInputOffset = prevLayer.ActivationOutputOffset;
        ActivationOutputOffset = prevLayer.ActivationOutputOffset + prevLayer.NumOutputs;
        CurrentLayerErrorOffset = prevLayer.NextLayerErrorOffset;
        NextLayerErrorOffset = prevLayer.CurrentLayerErrorOffset + NumInputs;
    }

    internal override void Connect(IInputConfig config)
    {
        if (config is not ConnectedInputConfig c) throw new Exception();

        NumOutputs = Neurons;
        NumInputs = c.NumInputs;
        BiasOffset = NumInputs * NumOutputs;

        ParameterCount = NumOutputs * NumInputs + NumOutputs;
        ParameterOffset = 0;

        ActivationInputOffset = 0;
        ActivationOutputOffset = c.NumInputs;
        CurrentLayerErrorOffset = 0;
        NextLayerErrorOffset = c.NumInputs;
    }
}

public class Dropout : ConnectedLayer
{
    public float DropoutRate { get; set; }

    internal override void Connect(Layer prevLayer)
    {
        NumInputs = NumOutputs = prevLayer.NumOutputs;

        ParameterCount = 0;
        ParameterOffset = prevLayer.ParameterOffset + prevLayer.ParameterCount;

        ActivationInputOffset = prevLayer.ActivationOutputOffset;
        ActivationOutputOffset = prevLayer.ActivationOutputOffset + prevLayer.NumOutputs;
        CurrentLayerErrorOffset = prevLayer.NextLayerErrorOffset;
        NextLayerErrorOffset = prevLayer.CurrentLayerErrorOffset + NumInputs;
    }

    internal override void Connect(IInputConfig config)
    {
        if (config is not ConnectedInputConfig c) throw new Exception();

        NumInputs = NumOutputs = c.NumInputs;

        ParameterCount = 0;
        ParameterOffset = 0;

        ActivationInputOffset = 0;
        ActivationOutputOffset = c.NumInputs;
        CurrentLayerErrorOffset = 0;
        NextLayerErrorOffset = c.NumInputs;
    }
}

public class Softmax : ConnectedLayer
{
    public int Neurons { get; set; }
    public int BiasOffset { get; set; }

    internal override void Connect(Layer prevLayer)
    {
        NumOutputs = Neurons;
        NumInputs = prevLayer.NumOutputs;

        ParameterCount = NumOutputs * NumInputs + NumOutputs;
        ParameterOffset = prevLayer.ParameterOffset + prevLayer.ParameterCount;
        BiasOffset = NumInputs * NumOutputs;

        ActivationInputOffset = prevLayer.ActivationOutputOffset;
        ActivationOutputOffset = prevLayer.ActivationOutputOffset + prevLayer.NumOutputs;
        CurrentLayerErrorOffset = prevLayer.NextLayerErrorOffset;
        NextLayerErrorOffset = prevLayer.CurrentLayerErrorOffset + NumInputs;
    }

    internal override void Connect(IInputConfig config)
    {
        if (config is not ConnectedInputConfig c) throw new Exception();

        NumOutputs = Neurons;
        NumInputs = c.NumInputs;
        BiasOffset = NumInputs * NumOutputs;

        ParameterCount = NumOutputs * NumInputs + NumOutputs;
        ParameterOffset = 0;

        ActivationInputOffset = 0;
        ActivationOutputOffset = c.NumInputs;
        CurrentLayerErrorOffset = 0;
        NextLayerErrorOffset = c.NumInputs;
    }
}

public class Convolution : ConvolutionalLayer
{
    public int Stride { get; set; }
    public int Padding { get; set; }
    public int KernelsPerChannel { get; set; }
    public int KernelSize { get; set; }
    public ActivationType Activation { get; set; }

    internal override void Connect(IInputConfig config)
    {
        if (config is not ConvolutionInputConfig c) throw new Exception();

        InputChannels = c.Grayscale ? 1 : 3;
        InputWidth = c.Width;
        InputHeight = c.Height;
        NumInputs = InputWidth * InputHeight * InputChannels;
        OutputWidth = (InputWidth - KernelSize + 2 * Padding) / Stride + 1;
        OutputHeight = (InputHeight - KernelSize + 2 * Padding) / Stride + 1;
        OutputChannels = KernelsPerChannel * InputChannels;

        NumOutputs = OutputWidth * OutputHeight * OutputChannels;

        ParameterCount = OutputChannels * KernelSize * KernelSize;
        ParameterOffset = 0;

        ActivationInputOffset = 0;
        ActivationOutputOffset = NumInputs;
        CurrentLayerErrorOffset = 0;
        NextLayerErrorOffset = NumInputs;
    }

    internal override void Connect(Layer prevLayer)
    {
        if (prevLayer is not ConvolutionalLayer l) throw new Exception();

        InputChannels = l.OutputChannels;
        InputWidth = l.OutputWidth;
        InputHeight = l.OutputHeight;

        NumInputs = InputWidth * InputHeight * InputChannels;
        OutputWidth = (InputWidth - KernelSize + 2 * Padding) / Stride + 1;
        OutputHeight = (InputHeight - KernelSize + 2 * Padding) / Stride + 1;
        OutputChannels = KernelsPerChannel * InputChannels;
        NumOutputs = OutputWidth * OutputHeight * OutputChannels;

        ParameterCount = OutputChannels * KernelSize * KernelSize;
        ParameterOffset = l.ParameterOffset + l.ParameterCount;

        ActivationInputOffset = l.ActivationOutputOffset;
        ActivationOutputOffset = l.ActivationOutputOffset + l.NumOutputs;
        CurrentLayerErrorOffset = l.NextLayerErrorOffset;
        NextLayerErrorOffset = l.CurrentLayerErrorOffset + NumInputs;
    }
}

public class Maxpool : ConvolutionalLayer
{
    public int PoolSize { get; set; }

    internal override void Connect(Layer prevLayer)
    {
        if (prevLayer is not ConvolutionalLayer l) throw new Exception();

        InputWidth = l.OutputWidth;
        InputHeight = l.OutputHeight;
        OutputChannels = InputChannels = l.OutputChannels;

        OutputHeight = InputHeight / PoolSize;
        OutputWidth = InputWidth / PoolSize;
        NumInputs = InputWidth * InputHeight * InputChannels;
        NumOutputs = OutputWidth * OutputHeight * OutputChannels;

        ParameterCount = 0;
        ParameterOffset = prevLayer.ParameterOffset + prevLayer.ParameterCount;

        ActivationInputOffset = prevLayer.ActivationOutputOffset;
        ActivationOutputOffset = prevLayer.ActivationOutputOffset + prevLayer.NumOutputs;
        CurrentLayerErrorOffset = prevLayer.NextLayerErrorOffset;
        NextLayerErrorOffset = prevLayer.CurrentLayerErrorOffset + NumInputs;
    }

    internal override void Connect(IInputConfig config)
    {
        if (config is not ConvolutionInputConfig c) throw new Exception();

        InputWidth = c.Width;
        InputHeight = c.Height;
        OutputChannels = InputChannels = c.Grayscale ? 1 : 3;

        OutputHeight = InputHeight / PoolSize;
        OutputWidth = InputWidth / PoolSize;
        NumInputs = InputWidth * InputHeight * InputChannels;
        NumOutputs = OutputWidth * OutputHeight * OutputChannels;

        ParameterCount = 0;
        ParameterOffset = 0;

        ActivationInputOffset = 0;
        ActivationOutputOffset = NumInputs;
        CurrentLayerErrorOffset = 0;
        NextLayerErrorOffset = NumInputs;
    }
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