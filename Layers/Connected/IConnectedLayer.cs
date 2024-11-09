namespace vcortex.Layers.Connected;

public interface IConnectedLayer : ILayer
{
    public void Connect(ILayer prevLayer);
    public void Connect(ConnectedInputConfig config);
}