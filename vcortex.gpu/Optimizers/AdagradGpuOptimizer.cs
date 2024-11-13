using ILGPU;
using ILGPU.Runtime;
using vcortex.Core;
using vcortex.Core.Optimizers;

namespace vcortex.gpu.Optimizers;

public class AdagradOptimizer : IOptimizer
{
    private MemoryBuffer1D<float, Stride1D.Dense> _accumulatedSquares;
    private Action<Index1D, OptimizerKernelInput, ArrayView<float>, ArrayView<float>, ArrayView<float>> _optimizerKernel;

    public struct OptimizerKernelInput
    {
        public int BatchSize { get; set; }
        public int ParameterCount { get; set; }
        public float Epsilon { get; set; }
        public float LearningRate { get; set; }
    }
    private readonly AdaGrad _adaGrad;
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
    
    private OptimizerKernelInput _optimizerKernelInput;

    public AdagradOptimizer(AdaGrad adaGrad)
    {
        _adaGrad = adaGrad;
    }

    public void Compile(NetworkTrainer trainer)
    {
        _optimizerKernelInput = new OptimizerKernelInput()
        {
            BatchSize = trainer.Buffers.BatchSize,
            Epsilon = _adaGrad.Epsilon,
            ParameterCount = trainer.Network.ParameterCount,
            LearningRate = 0.1f
        };
        _accumulatedSquares = trainer.Accelerator.Allocate1D<float>(trainer.Network.ParameterCount);
        _optimizerKernel =
            trainer.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, OptimizerKernelInput, ArrayView<float>, ArrayView<float>, ArrayView<float>>(AdagradOptimizerKernelImpl);
    }

    public void Optimize(NetworkData networkData, NetworkAcceleratorBuffers buffers, float learningRate)
    {
        _optimizerKernelInput.LearningRate = learningRate;
        _optimizerKernel(networkData.ParameterCount, _optimizerKernelInput, buffers.Parameters.View, buffers.Gradients.View, _accumulatedSquares.View);
    }

    public void Dispose() { _accumulatedSquares.Dispose(); }
}