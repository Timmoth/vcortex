using vcortex.Network;
using vcortex.Optimizers;

namespace vcortex.cpu.Optimizers;

public class SGDWithMomentumOptimizer : IOptimizer
{
    private readonly SgdMomentum _sgdMomentum;
    private readonly NetworkAcceleratorBuffers _buffers;
    private readonly NetworkData _networkData;
    
    private readonly float[] _velocity;

    public SGDWithMomentumOptimizer(SgdMomentum sgdMomentum, NetworkAcceleratorBuffers buffers, NetworkData networkData)
    {
        _sgdMomentum = sgdMomentum;
        _buffers = buffers;
        _networkData = networkData;

        _velocity = new float[networkData.ParameterCount];
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

            // Update the momentum term
            _velocity[parameterIndex] = _sgdMomentum.Momentum * _velocity[parameterIndex] + (1 - _sgdMomentum.Momentum) * gradientSum;

            // Update the parameter using SGD with momentum update rule
            _buffers.Parameters[parameterIndex] -= learningRate * _velocity[parameterIndex];
        });
    }
}