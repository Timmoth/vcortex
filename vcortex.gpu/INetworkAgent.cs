using ILGPU.Runtime;
using vcortex.Network;

namespace vcortex.gpu;

public interface INetworkAgent : IDisposable
{
    Accelerator Accelerator { get; }
    NetworkConfig Network { get; }
    NetworkAcceleratorBuffers Buffers { get; }

    bool IsTraining { get; }
}