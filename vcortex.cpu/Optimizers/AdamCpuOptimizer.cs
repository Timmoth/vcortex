using vcortex.Network;
using vcortex.Optimizers;

namespace vcortex.cpu.Optimizers;

public class AdamOptimizer : IOptimizer
{
    private readonly Adam _adam;
    public int Timestep;

    private readonly NetworkAcceleratorBuffers _buffers;
    private readonly NetworkData _networkData;
    private readonly float[] _firstMoment;
    private readonly float[] _secondMoment;

    public AdamOptimizer(Adam adam, NetworkAcceleratorBuffers buffers, NetworkData networkData)
    {
        _adam = adam;
        _buffers = buffers;
        _networkData = networkData;
        _firstMoment = new float[networkData.ParameterCount];
        _secondMoment = new float[networkData.ParameterCount];
    }

    public void Optimize(float learningRate)
    {
        Timestep++;
        var biasCorrection1 = 1 - MathF.Pow(_adam.Beta1, Timestep);
        var biasCorrection2 = 1 - MathF.Pow(_adam.Beta2, Timestep);
        
        Parallel.For(0, _networkData.ParameterCount, (parameterIndex) =>
        {
            // Accumulate gradients across the batch for each parameter
            var gradientSum = 0.0f;
            for (var i = 0; i < _buffers.BatchSize; i++) gradientSum += _buffers.Gradients[i * _networkData.ParameterCount + parameterIndex];

            // Average the gradient over the batch size
            gradientSum /= _buffers.BatchSize;

            // Update the first and second moment estimates for the Adam optimizer
            _firstMoment[parameterIndex] =
                _adam.Beta1 * _firstMoment[parameterIndex] + (1 - _adam.Beta1) * gradientSum;
            _secondMoment[parameterIndex] = _adam.Beta2 * _secondMoment[parameterIndex] +
                                            (1 - _adam.Beta2) * gradientSum * gradientSum;

            // Apply bias correction for the moments
            var mHat = _firstMoment[parameterIndex] / biasCorrection1;
            var vHat = _secondMoment[parameterIndex] / biasCorrection2;

            // Update parameter using Adam optimizer formula
            _buffers.Parameters[parameterIndex] -= learningRate * mHat / (MathF.Sqrt(vHat) + _adam.Epsilon);
        });
        
    }

    public void Reset()
    {
        Timestep = 0;
    }

    public void Dispose()
    {
    }
}