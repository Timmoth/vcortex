using ILGPU.Runtime;

namespace vcortex.gpu;

public interface INetworkAgent : IDisposable
{
    Accelerator Accelerator { get; }
    Network Network { get; }
    NetworkAcceleratorBuffers Buffers { get; }
    
    bool IsTraining { get; }
}