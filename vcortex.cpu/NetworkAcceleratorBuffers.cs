using vcortex.Network;

namespace vcortex.cpu;

public class NetworkAcceleratorBuffers : IDisposable
{
    public readonly float[] Activations;
    public readonly int BatchSize;
    public readonly float[] Errors;
    public readonly float[] Gradients;
    public readonly float[] Parameters;

    public NetworkAcceleratorBuffers(NetworkConfig network, int batchSize)
    {
        BatchSize = batchSize;
        Parameters = new float[network.NetworkData.ParameterCount];
        Activations = new float[network.NetworkData.ActivationCount * batchSize];
        Gradients = new float[network.NetworkData.ParameterCount * batchSize];
        Errors = new float[network.NetworkData.ActivationCount * batchSize];
    }

    public void Dispose()
    {

    }
}