using vcortex.Layers;

namespace vcortex.cpu.Layers;

public interface ILayer
{
    public Layer Config { get; }
    public void FillRandom(INetworkAgent agent);

    public void Forward(INetworkAgent agent);
    public void Backward(NetworkTrainer trainer);
}