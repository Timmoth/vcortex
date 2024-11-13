using ILGPU;
using ILGPU.Runtime;
using vcortex.Network;
using vcortex.Optimizers;

namespace vcortex.gpu.Optimizers;

public class RMSpropOptimizer : IOptimizer
{
    private readonly RmsProp _rmsProp;
    private MemoryBuffer1D<float, Stride1D.Dense> _movingAverageOfSquares;

    private Action<Index1D, OptimizerKernelInput, ArrayView<float>, ArrayView<float>, ArrayView<float>>
        _optimizerKernel;

    private OptimizerKernelInput _optimizerKernelInput;

    public RMSpropOptimizer(RmsProp rmsProp)
    {
        _rmsProp = rmsProp;
    }

    public void Compile(NetworkTrainer trainer)
    {
        _optimizerKernelInput = new OptimizerKernelInput
        {
            BatchSize = trainer.Buffers.BatchSize,
            ParameterCount = trainer.Network.NetworkData.ParameterCount,
            Epsilon = _rmsProp.Epsilon,
            Rho = _rmsProp.Rho,
            LearningRate = 0.01f
        };

        _movingAverageOfSquares = trainer.Accelerator.Allocate1D<float>(trainer.Network.NetworkData.ParameterCount);
        _optimizerKernel =
            trainer.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, OptimizerKernelInput, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>>(RMSpropOptimizerKernelImpl);
    }

    public void Optimize(NetworkData networkData, NetworkAcceleratorBuffers buffers, float learningRate)
    {
        _optimizerKernelInput.LearningRate = learningRate;
        _optimizerKernel(networkData.ParameterCount, _optimizerKernelInput, buffers.Parameters.View,
            buffers.Gradients.View, _movingAverageOfSquares.View);
    }

    public void Dispose()
    {
        _movingAverageOfSquares.Dispose();
    }

    public static void RMSpropOptimizerKernelImpl(
        Index1D index,
        OptimizerKernelInput input,
        ArrayView<float> parameters,
        ArrayView<float> gradients,
        ArrayView<float> movingAverageOfSquares)
    {
        var batchSize = input.BatchSize;

        // Accumulate gradients across the batch for each parameter
        var gradientSum = 0.0f;
        for (var i = 0; i < batchSize; i++) gradientSum += gradients[i * input.ParameterCount + index];

        // Average the gradient over the batch size
        gradientSum /= batchSize;

        // Update moving average of squared gradients
        movingAverageOfSquares[index] = input.Rho * movingAverageOfSquares[index] +
                                        (1 - input.Rho) * gradientSum * gradientSum;

        // Update parameter using RMSprop update rule
        parameters[index] -= input.LearningRate * gradientSum /
                             (MathF.Sqrt(movingAverageOfSquares[index]) + input.Epsilon);
    }

    public struct OptimizerKernelInput
    {
        public int BatchSize { get; set; }
        public int ParameterCount { get; set; }
        public float Rho { get; set; }
        public float Epsilon { get; set; }
        public float LearningRate { get; set; }
    }
}