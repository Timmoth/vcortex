using ILGPU.Runtime;
using vcortex.Network;

namespace vcortex.gpu;

public interface IGpuNetworkAgent : IDisposable
{
    Accelerator Accelerator { get; }
    NetworkConfig Network { get; }
    NetworkAcceleratorBuffers Buffers { get; }

    bool IsTraining { get; }

    public void SaveParametersToDisk(string filePath);
    public void ReadParametersFromDisk(string filePath);
    public float[] GetParameters();
    public void LoadParameters(float[] parameters);
}