using vcortex.Network;
using vcortex.Optimizers;

namespace vcortex.cpu.Optimizers;

public class RMSpropOptimizer : IOptimizer
{
    private readonly RmsProp _rmsProp;
    private readonly NetworkBuffers _buffers;
    private readonly NetworkData _networkData;
    
    private readonly float[] _movingAverageOfSquares;

    public RMSpropOptimizer(RmsProp rmsProp, NetworkBuffers buffers, NetworkData networkData)
    {
        _rmsProp = rmsProp;
        _buffers = buffers;
        _networkData = networkData;

        _movingAverageOfSquares = new float[networkData.ParameterCount];
    }
    public void Dispose()
    {
        throw new NotImplementedException();
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

            // Update moving average of squared gradients
            _movingAverageOfSquares[parameterIndex] = _rmsProp.Rho * _movingAverageOfSquares[parameterIndex] +
                                                      (1 - _rmsProp.Rho) * gradientSum * gradientSum;

            // Update parameter using RMSprop update rule
            _buffers.Parameters[parameterIndex] -= learningRate * gradientSum /
                                                  (MathF.Sqrt(_movingAverageOfSquares[parameterIndex]) + _rmsProp.Epsilon);
        });
        
    }
}