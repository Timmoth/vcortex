using vcortex.Network;

namespace vcortex.cpu.LossFunctions;

public class CpuCrossEntropyLoss : ILossFunction
{
    private readonly NetworkBuffers _buffers;
    private readonly NetworkConfig _network;

    public CpuCrossEntropyLoss(NetworkBuffers buffers, NetworkConfig network)
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
                var expected = expectedOutputs[outputIndex];
                var actual = _buffers.Activations[activationOutputIndex + outputIndex];
                
                // Compute Cross-Entropy loss for each output (assuming outputs are one-hot encoded)
                // We use a small epsilon to prevent log(0)
                var epsilon = 1e-15f;
                var logProb = MathF.Max(actual, epsilon); // Log of the predicted probability (softmax output)

                // Compute the loss for the current sample
                var loss = -expected * MathF.Log(logProb);

                // Calculate the gradient of the loss w.r.t. the predicted probability (backpropagation)
                // Derivative of cross-entropy loss with softmax is: p - y
                var error = actual - expected;

                // Store the gradient in the errors array
                _buffers.Errors[nextErrorOffset + outputIndex] = error;
                sampleError += loss;
            }
        }

        return sampleError / finalLayer.NumOutputs;
    }
}