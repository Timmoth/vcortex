using vcortex.Network;

namespace vcortex.cpu;

public interface INetworkAgent : IDisposable
{
    NetworkConfig Network { get; }
    NetworkAcceleratorBuffers Buffers { get; }

    bool IsTraining { get; }
}