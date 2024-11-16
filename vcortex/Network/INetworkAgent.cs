namespace vcortex.Network;

public interface INetworkAgent : IDisposable
{
    public void SaveParametersToDisk(string filePath);
    public void ReadParametersFromDisk(string filePath);
    public float[] GetParameters();
    public void LoadParameters(float[] parameters);
    void InitRandomParameters();
}