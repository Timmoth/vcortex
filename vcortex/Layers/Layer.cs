using vcortex.Input;

namespace vcortex.Layers;

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