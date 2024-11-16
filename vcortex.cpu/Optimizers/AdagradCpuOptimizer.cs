using vcortex.Network;
using vcortex.Optimizers;

namespace vcortex.cpu.Optimizers;

public class AdagradOptimizer : IOptimizer
{
    private readonly AdaGrad _adaGrad;
    private readonly NetworkBuffers _buffers;
    private readonly NetworkData _networkData;
    private readonly float[] _accumulatedSquares;
    public AdagradOptimizer(AdaGrad adaGrad, NetworkBuffers buffers, NetworkData networkData)
    {
        _adaGrad = adaGrad;
        _buffers = buffers;
        _networkData = networkData;
        _accumulatedSquares = new float[networkData.ParameterCount];
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

            // Update accumulated squared gradients
            _accumulatedSquares[parameterIndex] += gradientSum * gradientSum;

            // Update parameter using Adagrad update rule
            _buffers.Parameters[parameterIndex] -= learningRate * gradientSum / (MathF.Sqrt(_accumulatedSquares[parameterIndex]) + _adaGrad.Epsilon);
        });
        
    }
}