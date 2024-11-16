using vcortex.Network;

namespace vcortex.cpu.LossFunctions;

public class CpuMseLoss : ILossFunction
{
    private readonly NetworkBuffers _buffers;
    private readonly NetworkConfig _network;

    public CpuMseLoss(NetworkBuffers buffers, NetworkConfig network)
    {
        _buffers = buffers;
        _network = network;
    }


    public void Dispose()
    {
        
    }

    public float Apply(List<(float[] inputs, float[] expectedOutputs)> batch)
    {
        Array.Clear(_buffers.Errors);
        var finalLayer = _network.Layers[^1];
        var sampleError = 0.0f;
        for (var batchIndex = 0; batchIndex < batch.Count; batchIndex++)
        {
            var activationOutputIndex = batchIndex * _network.NetworkData.ActivationCount + finalLayer.ActivationOutputOffset;
            var nextErrorOffset = batchIndex * _network.NetworkData.ActivationCount + finalLayer.NextLayerErrorOffset;

            var expectedOutputs = batch[batchIndex].expectedOutputs;
            for (var outputIndex = 0; outputIndex < finalLayer.NumOutputs; outputIndex++)
            {
                var error = _buffers.Activations[activationOutputIndex + outputIndex] - expectedOutputs[outputIndex];
                _buffers.Errors[nextErrorOffset + outputIndex] = error;
                sampleError += error * error;
            }
        }

        return sampleError / finalLayer.NumOutputs;
    }
}