using vcortex.Layers;
using vcortex.Network;

namespace vcortex.cpu.Layers;

public class DenseLayer : IConnectedLayer
{
    private readonly Dense _dense;
    private readonly NetworkBuffers _buffers;
    private readonly NetworkData _networkData;
    public DenseLayer(Dense dense, NetworkBuffers buffers, NetworkData networkData)
    {
        _dense = dense;
        _buffers = buffers;
        _networkData = networkData;
    }

    public Layer Config => _dense;
    public void FillRandom()
    {
        var rnd = Random.Shared;
        var variance = 2.0f / _dense.ParameterCount;
        for (var i = 0; i < _dense.ParameterCount; i++)
            _buffers.Parameters[_dense.ParameterOffset + i] = (float)(rnd.NextDouble() * 2 - 1) * MathF.Sqrt(variance);
    }

    public void Forward()
    {
        Parallel.For(0, _buffers.BatchSize, batchIndex =>
        {
            // Compute activation offsets once per batch
            var activationInputOffset = batchIndex * _networkData.ActivationCount + _dense.ActivationInputOffset;
            var activationOutputOffset = batchIndex * _networkData.ActivationCount + _dense.ActivationOutputOffset;

            for (var outputIndex = 0; outputIndex < _dense.NumOutputs; outputIndex++)
            {
                // Offset for weights of the current output neuron
                var weightOffset = _dense.ParameterOffset + outputIndex * _dense.NumInputs;

                // Start sum with bias, computed once per output neuron
                var sum = _buffers.Parameters[_dense.ParameterOffset + _dense.BiasOffset + outputIndex];

                // Accumulate weighted input activations
                for (var j = 0; j < _dense.NumInputs; j++)
                {
                    sum += _buffers.Activations[activationInputOffset + j] * 
                           _buffers.Parameters[weightOffset + j];
                }

                if (_dense.Activation == ActivationType.Sigmoid)
                    // Sigmoid activation function: 1 / (1 + e^(-x))
                    _buffers.Activations[activationOutputOffset + outputIndex] = 1.0f / (1.0f + float.Exp(-sum));
                else if (_dense.Activation == ActivationType.Relu)
                    // ReLU activation function: max(0, x)
                    _buffers.Activations[activationOutputOffset + outputIndex] = float.Max(0.0f, sum);
                else if (_dense.Activation == ActivationType.LeakyRelu)
                    // Leaky ReLU activation function: max(alpha * x, x)
                    _buffers.Activations[activationOutputOffset + outputIndex] = float.Max(0.1f * sum, sum);

            }
        });
    }


    public void Backward()
    {
         Parallel.For(0, _buffers.BatchSize, batchIndex =>
        {
            // Calculate relevant offsets
            var activationInputOffset = batchIndex * _networkData.ActivationCount + _dense.ActivationInputOffset;
            var activationOutputOffset = batchIndex * _networkData.ActivationCount + _dense.ActivationOutputOffset;
            var currentErrorOffset = batchIndex * _networkData.ActivationCount + _dense.CurrentLayerErrorOffset;
            var nextErrorOffset = batchIndex * _networkData.ActivationCount + _dense.NextLayerErrorOffset;
            var gradientOffset = batchIndex * _networkData.ParameterCount + _dense.ParameterOffset;
            
            for (var outputIndex = 0; outputIndex < _dense.NumOutputs; outputIndex++)
            {
                for (var inputIndex = 0; inputIndex < _dense.NumInputs; inputIndex++)
                {
                    // Calculate the derivative of the sigmoid activation
                    var x = _buffers.Activations[activationOutputOffset + outputIndex];
                    var derivative = 0.0f;
                    if (_dense.Activation == ActivationType.Sigmoid)
                        // Sigmoid derivative: x * (1 - x) where x is the sigmoid output
                        derivative = x * (1.0f - x);
                    else if (_dense.Activation == ActivationType.Relu)
                        // ReLU derivative: 1 if x > 0, else 0
                        derivative = x > 0 ? 1.0f : 0.0f;
                    else if (_dense.Activation == ActivationType.LeakyRelu)
                        // Leaky ReLU derivative: 1 if x > 0, else alpha
                        derivative = x > 0 ? 1.0f : 0.01f;

                    var delta = _buffers.Errors[nextErrorOffset + outputIndex] * derivative;

                    // Compute the offset for weights of this output neuron
                    var weightIndex = outputIndex * _dense.NumInputs + inputIndex;

                    // Atomic update to propagate error back to current layer and accumulate weight gradient
                    _buffers.Errors[currentErrorOffset + inputIndex] +=
                        delta * _buffers.Parameters[_dense.ParameterOffset + weightIndex];
                    _buffers.Gradients[gradientOffset + weightIndex] = delta * _buffers.Activations[activationInputOffset + inputIndex];
                }
            }
        });
        
         Parallel.For(0, _buffers.BatchSize, batchIndex =>
        {
            // Calculate relevant offsets
            var activationOutputOffset = batchIndex * _networkData.ActivationCount + _dense.ActivationOutputOffset;
            var nextErrorOffset = batchIndex * _networkData.ActivationCount + _dense.NextLayerErrorOffset;
            var gradientOffset = batchIndex * _networkData.ParameterCount + _dense.ParameterOffset;
            
            
            for (var outputIndex = 0; outputIndex < _dense.NumOutputs; outputIndex++)
            {
                // Compute delta using sigmoid derivative
                var x = _buffers.Activations[activationOutputOffset + outputIndex];
                var derivative = 0.0f;
                if (_dense.Activation == ActivationType.Sigmoid)
                    // Sigmoid derivative: x * (1 - x) where x is the sigmoid output
                    derivative = x * (1.0f - x);
                else if (_dense.Activation == ActivationType.Relu)
                    // ReLU derivative: 1 if x > 0, else 0
                    derivative = x > 0 ? 1.0f : 0.0f;
                else if (_dense.Activation == ActivationType.LeakyRelu)
                    // Leaky ReLU derivative: 1 if x > 0, else alpha
                    derivative = x > 0 ? 1.0f : 0.01f;

                var delta = _buffers.Errors[nextErrorOffset + outputIndex] * derivative;

                // Update bias gradient for this neuron
                _buffers.Gradients[gradientOffset + _dense.BiasOffset + outputIndex] = delta;
            }
            
        });
         
    }

    public bool IsTraining { get; set; }
}