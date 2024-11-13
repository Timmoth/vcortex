using ILGPU.Runtime;
using vcortex.Core.Layers;

namespace vcortex.gpu;

public interface INetworkAgent : IDisposable
{
    Accelerator Accelerator { get; }
    NetworkConfig Network { get; }
    NetworkAcceleratorBuffers Buffers { get; }

    bool IsTraining { get; }
}