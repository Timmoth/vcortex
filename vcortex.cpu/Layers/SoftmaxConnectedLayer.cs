using vcortex.Layers;
using vcortex.Network;

namespace vcortex.cpu.Layers;

public class SoftmaxConnectedLayer : IConnectedLayer
{
    private readonly Softmax _softmax;
    private readonly NetworkBuffers _buffers;
    private readonly NetworkData _networkData;
    public SoftmaxConnectedLayer(Softmax softmax, NetworkBuffers buffers, NetworkData networkData)
    {
        _softmax = softmax;
        _buffers = buffers;
        _networkData = networkData;
    }

    public Layer Config => _softmax;
    public void FillRandom()
    {
        var rnd = Random.Shared;
        var variance = 2.0f / _softmax.ParameterCount;
        for (var i = 0; i < _softmax.ParameterCount; i++)
            _buffers.Parameters[_softmax.ParameterOffset +i] = (float)(rnd.NextDouble() * 2 - 1) * MathF.Sqrt(variance);
    }

 public void Forward()
{
    Parallel.For(0, _buffers.BatchSize, batchIndex =>
    {
        // Calculate common offsets once per batch
        var activationInputOffset = batchIndex * _networkData.ActivationCount + _softmax.ActivationInputOffset;
        var activationOutputOffset = batchIndex * _networkData.ActivationCount + _softmax.ActivationOutputOffset;
        var weightOffsetBase = _softmax.ParameterOffset;

        // First pass: Compute raw scores (weighted sum of inputs + bias)
        for (var outputIndex = 0; outputIndex < _softmax.NumOutputs; outputIndex++)
        {
            // Initialize the sum with the bias term
            var sum = _buffers.Parameters[weightOffsetBase + _softmax.BiasOffset + outputIndex];

            // Compute the weighted sum for this output
            var weightsOffset = weightOffsetBase + _softmax.NumInputs * outputIndex;
            for (var j = 0; j < _softmax.NumInputs; j++)
                sum += _buffers.Activations[activationInputOffset + j] * _buffers.Parameters[weightsOffset + j];

            // Store the result (raw score) in the output activations
            _buffers.Activations[activationOutputOffset + outputIndex] = sum;
        }

        // Second pass: Apply softmax normalization
        // Find the maximum value in the outputs for numerical stability
        var maxVal = _buffers.Activations[activationOutputOffset];
        for (var i = 1; i < _softmax.NumOutputs; i++)
            maxVal = Math.Max(maxVal, _buffers.Activations[activationOutputOffset + i]);

        // Compute the sum of exponentials and apply softmax
        float sumExp = 0;
        for (var i = 0; i < _softmax.NumOutputs; i++)
        {
            // Calculate exp(x - max(x)) for numerical stability
            var expValue = float.Exp(_buffers.Activations[activationOutputOffset + i] - maxVal);
            _buffers.Activations[activationOutputOffset + i] = expValue;

            sumExp += expValue;
        }

        // If sumExp is invalid (zero, NaN, or Infinity), assign uniform probabilities
        if (sumExp <= 0 || float.IsNaN(sumExp) || float.IsInfinity(sumExp))
        {
            var uniformProb = 1.0f / _softmax.NumOutputs;
            for (var i = 0; i < _softmax.NumOutputs; i++)
                _buffers.Activations[activationOutputOffset + i] = uniformProb;
        }
        else
        {
            // Normalize to get probabilities
            for (var i = 0; i < _softmax.NumOutputs; i++)
                _buffers.Activations[activationOutputOffset + i] /= sumExp;
        }
    });
}

    public void Backward()
    {
        Parallel.For(0, _buffers.BatchSize, batchIndex =>
        {
            // Offsets for activations and errors
            var activationInputOffset = batchIndex * _networkData.ActivationCount + _softmax.ActivationInputOffset;
            var currentErrorOffset = batchIndex * _networkData.ActivationCount + _softmax.CurrentLayerErrorOffset;
            var nextErrorOffset = batchIndex * _networkData.ActivationCount + _softmax.NextLayerErrorOffset;
            var gradientOffset = batchIndex * _networkData.ParameterCount + _softmax.ParameterOffset;
            
            for (int outputIndex = 0; outputIndex < _softmax.NumOutputs; outputIndex++)
            {
                for (int inputIndex = 0; inputIndex < _softmax.NumInputs; inputIndex++)
                {
                    // Calculate delta for this output neuron
                    var delta = _buffers.Errors[nextErrorOffset + outputIndex];

                    // Compute the gradient contribution for each weight and accumulate errors for backpropagation
                    _buffers.Errors[currentErrorOffset + inputIndex] +=
                        delta * _buffers.Parameters[_softmax.ParameterOffset + _softmax.NumInputs * outputIndex + inputIndex];

                    // Store gradient for the current weight
                    _buffers.Gradients[gradientOffset + outputIndex * _softmax.NumInputs + inputIndex] =
                        delta * _buffers.Activations[activationInputOffset + inputIndex];
                }
            }
        });
        
        Parallel.For(0, _buffers.BatchSize, batchIndex =>
        {
            var nextErrorOffset = batchIndex * _networkData.ActivationCount + _softmax.NextLayerErrorOffset;
            var gradientOffset = batchIndex * _networkData.ParameterCount + _softmax.ParameterOffset;
            
            for (int outputIndex = 0; outputIndex < _softmax.NumOutputs; outputIndex++)
            {
                // Calculate gradient for the bias term of this output neuron
                var delta = _buffers.Errors[nextErrorOffset + outputIndex];
                _buffers.Gradients[gradientOffset + _softmax.BiasOffset + outputIndex] = delta;
            }
        });
    }

    public bool IsTraining { get; set; }
}