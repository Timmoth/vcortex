using vcortex.Layers;
using vcortex.Network;

namespace vcortex.cpu.Layers;

public class DropoutLayer : IConnectedLayer
{
    private readonly Dropout _dropout;
    private readonly NetworkBuffers _buffers;
    private readonly NetworkData _networkData;
    private readonly bool[] _mask;
    private readonly Random _random;
    public DropoutLayer(Dropout dropout, NetworkBuffers buffers, NetworkData networkData)
    {
        _dropout = dropout;
        _buffers = buffers;
        _networkData = networkData;
        _mask = new bool[_dropout.NumOutputs];
        _random = new Random();
    }

    public Layer Config => _dropout;
    public void FillRandom()
    {
        var rnd = Random.Shared;
        var variance = 2.0f / _dropout.ParameterCount;
        for (var i = 0; i < _dropout.ParameterCount; i++)
            _buffers.Parameters[_dropout.ParameterOffset +i] = (float)(rnd.NextDouble() * 2 - 1) * MathF.Sqrt(variance);
    }

    public void Forward()
    {
        for (var i = 0; i < _dropout.NumOutputs; i++)
        {
            _mask[i] = !IsTraining || _random.NextSingle() > _dropout.DropoutRate;
        }

        Parallel.For(0, _buffers.BatchSize, batchIndex =>
        {
            // Precompute offsets for this batch
            var activationInputOffset = batchIndex * _networkData.ActivationCount + _dropout.ActivationInputOffset;
            var activationOutputOffset = batchIndex * _networkData.ActivationCount + _dropout.ActivationOutputOffset;

            for (var outputIndex = 0; outputIndex < _dropout.NumOutputs; outputIndex++)
            {
                // Apply mask to input activations
                _buffers.Activations[activationOutputOffset + outputIndex] = _mask[outputIndex]
                    ? _buffers.Activations[activationInputOffset + outputIndex]
                    : 0.0f;
            }
        });
    }

    public void Backward()
    {
        Parallel.For(0, _buffers.BatchSize, batchIndex =>
        {
            // Precompute offsets for this batch
            var currentErrorOffset = batchIndex * _networkData.ActivationCount + _dropout.CurrentLayerErrorOffset;
            var nextErrorOffset = batchIndex * _networkData.ActivationCount + _dropout.NextLayerErrorOffset;
            
            for (var outputIndex = 0; outputIndex < _dropout.NumOutputs; outputIndex++)
            {
                // Only propagate errors for active neurons (those that were not dropped)
                _buffers.Errors[currentErrorOffset + outputIndex] =
                    _mask[outputIndex] ? _buffers.Errors[nextErrorOffset + outputIndex] : 0.0f;
            }
        });
    }

    public bool IsTraining { get; set; }

}