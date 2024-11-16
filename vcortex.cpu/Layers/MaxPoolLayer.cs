using vcortex.Layers;
using vcortex.Network;

namespace vcortex.cpu.Layers;

public class MaxPoolLayer : IConvolutionalLayer
{
    private readonly Maxpool _maxpool;
    private readonly NetworkBuffers _buffers;
    private readonly NetworkData _networkData;
    public MaxPoolLayer(Maxpool maxpool, NetworkBuffers buffers, NetworkData networkData)
    {
        _maxpool = maxpool;
        _buffers = buffers;
        _networkData = networkData;
    }

    public Layer Config => _maxpool;
    public void FillRandom()
    {
        var rnd = Random.Shared;
        var variance = 2.0f / _maxpool.ParameterCount;
        for (var i = 0; i < _maxpool.ParameterCount; i++)
            _buffers.Parameters[_maxpool.ParameterOffset +i] = (float)(rnd.NextDouble() * 2 - 1) * MathF.Sqrt(variance);
    }

 public void Forward()
{
    Parallel.For(0, _buffers.BatchSize, batchIndex =>
    {
        // Calculate batch-specific offsets just once
        var activationInputOffset = batchIndex * _networkData.ActivationCount + _maxpool.ActivationInputOffset;
        var activationOutputOffset = batchIndex * _networkData.ActivationCount + _maxpool.ActivationOutputOffset;

        // Channel and pooling region offsets precomputed
        var inputChannelOffset = _maxpool.InputWidth * _maxpool.InputHeight;
        var outputChannelOffset = _maxpool.OutputWidth * _maxpool.OutputHeight;

        for (var c = 0; c < _maxpool.InputChannels; c++)
        {
            for (var y = 0; y < _maxpool.OutputHeight; y++)
            {
                for (var x = 0; x < _maxpool.OutputWidth; x++)
                {
                    var outputIndex = y * _maxpool.OutputWidth + x + c * outputChannelOffset;
                    float max = float.MinValue;

                    // Compute the base input index for this position, reducing per-element calculations
                    var baseX = x * _maxpool.PoolSize;
                    var baseY = y * _maxpool.PoolSize;
                    var channelBaseOffset = activationInputOffset + c * inputChannelOffset;

                    // Traverse the pooling window and find the maximum activation
                    for (var ky = 0; ky < _maxpool.PoolSize; ky++)
                    {
                        var rowOffset = (baseY + ky) * _maxpool.InputWidth + channelBaseOffset;
                        for (var kx = 0; kx < _maxpool.PoolSize; kx++)
                        {
                            // Calculate input index within pooling window
                            var inputIndex = rowOffset + (baseX + kx);

                            // Update max if the current input activation is greater
                            max = Math.Max(max, _buffers.Activations[inputIndex]);
                        }
                    }

                    // Store the maximum value in the output activation array
                    _buffers.Activations[activationOutputOffset + outputIndex] = max;
                }
            }
        }
    });
}


    public void Backward()
    {
            Parallel.For(0, _buffers.BatchSize, batchIndex =>
    {
      
        for (var c = 0; c < _maxpool.InputChannels; c++)
        {
            for (var y = 0; y < _maxpool.OutputHeight; y++)
            {
                for (var x = 0; x < _maxpool.OutputWidth; x++)
                {
                    // Offsets for activations and errors in the current and next layers
                    var activationInputOffset = batchIndex * _networkData.ActivationCount + _maxpool.ActivationInputOffset;
                    var currentErrorOffset = batchIndex * _networkData.ActivationCount + _maxpool.CurrentLayerErrorOffset;
                    var nextErrorOffset = batchIndex * _networkData.ActivationCount + _maxpool.NextLayerErrorOffset;

                    // Offsets for input and output channels
                    var inputChannelOffset = _maxpool.InputWidth * _maxpool.InputHeight;
                    var outputChannelOffset = _maxpool.OutputWidth * _maxpool.OutputHeight;

                    // Index in the output error for the current pooling position
                    var outputIndex = y * _maxpool.OutputWidth + x + c * outputChannelOffset;

                    // Initialize max value to track the maximum and its position
                    var max = float.MinValue;
                    var maxIndex = 0;

                    // Traverse the pooling window to identify the maximum activation and its index
                    for (var ky = 0; ky < _maxpool.PoolSize; ky++)
                    for (var kx = 0; kx < _maxpool.PoolSize; kx++)
                    {
                        // Calculate input coordinates (oldX, oldY) for the pooling window
                        var oldX = x * _maxpool.PoolSize + kx;
                        var oldY = y * _maxpool.PoolSize + ky;

                        // Compute input index for this (oldY, oldX) within the channel
                        var inputIndex = oldY * _maxpool.InputWidth + oldX + c * inputChannelOffset;

                        // Update the maximum and record the index if a new max is found
                        if (_buffers.Activations[activationInputOffset + inputIndex] >= max)
                        {
                            max = _buffers.Activations[activationInputOffset + inputIndex];
                            maxIndex = inputIndex;
                        }
                    }

                    // propagate the error to the position where the max was found
                    if (maxIndex >= 0) _buffers.Errors[currentErrorOffset + maxIndex] = _buffers.Errors[nextErrorOffset + outputIndex];
                }
            }
        }
    });
            
    }

    public bool IsTraining { get; set; }
}