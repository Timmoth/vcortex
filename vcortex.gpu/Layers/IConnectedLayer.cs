using vcortex.Core;
using vcortex.Core.Layers;

namespace vcortex.gpu.Layers;

public interface IConnectedLayer : ILayer
{
    public void Connect(ILayer prevLayer);
    public void Connect(ConnectedInputConfig config);
}