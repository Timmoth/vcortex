using vcortex.Network;

namespace vcortex.cpu;

public interface ICpuNetworkAgent : IDisposable
{
    NetworkConfig Network { get; }
    NetworkAcceleratorBuffers Buffers { get; }

    bool IsTraining { get; }
    
    public void SaveParametersToDisk(string filePath);
    public void ReadParametersFromDisk(string filePath);
    public float[] GetParameters();
    public void LoadParameters(float[] parameters);
}