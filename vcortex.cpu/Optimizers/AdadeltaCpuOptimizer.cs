using vcortex.Network;
using vcortex.Optimizers;

namespace vcortex.cpu.Optimizers;

public class AdadeltaOptimizer : IOptimizer
{
    private readonly AdaDelta _adaDelta;

    private readonly NetworkBuffers _buffers;
    private readonly NetworkData _networkData;
    private readonly float[] _accumulatedGradients;
    private readonly float[] _accumulatedUpdates;
    
    public AdadeltaOptimizer(AdaDelta adaDelta, NetworkBuffers buffers, NetworkData networkData)
    {
        _adaDelta = adaDelta;
        _buffers = buffers;
        _networkData = networkData;

        _accumulatedGradients = new float[networkData.ParameterCount];
        _accumulatedUpdates = new float[networkData.ParameterCount];
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

            // Update accumulated gradients
            _accumulatedGradients[parameterIndex] =
                _adaDelta.Rho * _accumulatedGradients[parameterIndex] + (1 - _adaDelta.Rho) * gradientSum * gradientSum;

            // Compute update step
            var updateStep = MathF.Sqrt(_accumulatedUpdates[parameterIndex] + _adaDelta.Epsilon) /
                             MathF.Sqrt(_accumulatedGradients[parameterIndex] + _adaDelta.Epsilon);

            // Update accumulated updates
            _accumulatedUpdates[parameterIndex] = _adaDelta.Rho * _accumulatedUpdates[parameterIndex] + (1 - _adaDelta.Rho) * gradientSum * gradientSum;

            // Update parameter using Adadelta update rule
            _buffers.Parameters[parameterIndex] -= learningRate * updateStep;
        });
    }
}