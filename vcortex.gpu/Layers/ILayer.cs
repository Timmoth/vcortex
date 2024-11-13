using vcortex.Core;

namespace vcortex.gpu.Layers;

public interface ILayer
{
    public int NumInputs { get; }
    public int NumOutputs { get; }
    public int ActivationInputOffset { get; }
    public int ActivationOutputOffset { get; }
    public int CurrentLayerErrorOffset { get; }
    public int NextLayerErrorOffset { get; }
    public int ParameterCount { get; }
    public int ParameterOffset { get; }

    public LayerData LayerData { get; set; }

    public void FillRandom(INetworkAgent agent);

    public void Forward(INetworkAgent agent);
    public void Backward(NetworkTrainer trainer);

    public void CompileKernels(INetworkAgent agent);
}