using vcortex.Network;

namespace vcortex.cpu;

public class NetworkAcceleratorBuffers : IDisposable
{
    public readonly float[] Activations;
    public readonly int BatchSize;
    public readonly float[] Errors;
    public readonly float[] Gradients;
    public readonly float[] Inputs;
    public readonly float[] Outputs;
    public readonly float[] Parameters;

    public NetworkAcceleratorBuffers(NetworkConfig network, int batchSize)
    {
        BatchSize = batchSize;
        Parameters = new float[network.NetworkData.ParameterCount];
        Activations = new float[network.NetworkData.ActivationCount * batchSize];
        Gradients = new float[network.NetworkData.ParameterCount * batchSize];
        Errors = new float[network.NetworkData.ActivationCount * batchSize];
        Inputs = new float[network.Layers[0].NumInputs * batchSize];
        Outputs = new float[network.Layers[^1].NumOutputs * batchSize];
    }

    public void Dispose()
    {

    }
}