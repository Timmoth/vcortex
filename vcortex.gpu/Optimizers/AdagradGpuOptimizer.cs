using ILGPU;
using ILGPU.Runtime;
using vcortex.Optimizers;

namespace vcortex.gpu.Optimizers;

public class AdagradOptimizer : IOptimizer
{
    private readonly AdaGrad _adaGrad;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _accumulatedSquares;

    private readonly Action<Index1D, OptimizerKernelInput, ArrayView<float>, ArrayView<float>, ArrayView<float>>
        _optimizerKernel;

    private OptimizerKernelInput _optimizerKernelInput;

    private readonly NetworkTrainer _trainer;
    public AdagradOptimizer(AdaGrad adaGrad, NetworkTrainer trainer)
    {
        _adaGrad = adaGrad;
        _trainer = trainer;
        _optimizerKernelInput = new OptimizerKernelInput
        {
            BatchSize = trainer.Buffers.BatchSize,
            Epsilon = _adaGrad.Epsilon,
            ParameterCount = trainer.Network.NetworkData.ParameterCount,
            LearningRate = 0.1f
        };
        _accumulatedSquares = trainer.Accelerator.Allocate1D<float>(trainer.Network.NetworkData.ParameterCount);
        _optimizerKernel =
            trainer.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, OptimizerKernelInput, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>>(AdagradOptimizerKernelImpl);
    }

    public void Optimize(float learningRate)
    {
        _optimizerKernelInput.LearningRate = learningRate;
        _optimizerKernel(_trainer.Network.NetworkData.ParameterCount, _optimizerKernelInput, _trainer.Buffers.Parameters.View,
            _trainer.Buffers.Gradients.View, _accumulatedSquares.View);
    }

    public void Dispose()
    {
        _accumulatedSquares.Dispose();
    }

    public static void AdagradOptimizerKernelImpl(
        Index1D index,
        OptimizerKernelInput input,
        ArrayView<float> parameters,
        ArrayView<float> gradients,
        ArrayView<float> accumulatedSquares)
    {
        var batchSize = input.BatchSize;

        // Accumulate gradients across the batch for each parameter
        var gradientSum = 0.0f;
        for (var i = 0; i < batchSize; i++) gradientSum += gradients[i * input.ParameterCount + index];

        // Average the gradient over the batch size
        gradientSum /= batchSize;

        // Update accumulated squared gradients
        accumulatedSquares[index] += gradientSum * gradientSum;

        // Update parameter using Adagrad update rule
        parameters[index] -= input.LearningRate * gradientSum / (MathF.Sqrt(accumulatedSquares[index]) + input.Epsilon);
    }

    public struct OptimizerKernelInput
    {
        public int BatchSize { get; set; }
        public int ParameterCount { get; set; }
        public float Epsilon { get; set; }
        public float LearningRate { get; set; }
    }
}