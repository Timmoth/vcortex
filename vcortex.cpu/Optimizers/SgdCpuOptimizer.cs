using vcortex.Network;
using vcortex.Optimizers;

namespace vcortex.cpu.Optimizers;

public class SgdOptimizer : IOptimizer
{
    private readonly Sgd _sgd;
    private readonly NetworkAcceleratorBuffers _buffers;
    private readonly NetworkData _networkData;
    public SgdOptimizer(Sgd sgd, NetworkAcceleratorBuffers buffers, NetworkData networkData)
    {
        _sgd = sgd;
        _buffers = buffers;
        _networkData = networkData;
    }


    public void Dispose()
    {
    }

    public void Optimize(float learningRate)
    {
        Parallel.For(0, _networkData.ParameterCount, (parameterIndex) =>
        {
            // Accumulate gradients across the batch for each parameter
            var gradientSum = 0.0f;
            for (var i = 0; i < _buffers.BatchSize; i++) gradientSum += _buffers.Gradients[i * _networkData.ParameterCount + parameterIndex];

            // Average the gradient over the batch size
            gradientSum /= _buffers.BatchSize;

            // Update parameter using SGD update rule
            _buffers.Parameters[parameterIndex] -= learningRate * gradientSum;
        });
        
    }
}