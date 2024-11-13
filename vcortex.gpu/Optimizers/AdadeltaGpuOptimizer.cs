using ILGPU;
using ILGPU.Runtime;
using vcortex.Network;
using vcortex.Optimizers;

namespace vcortex.gpu.Optimizers;

public class AdadeltaOptimizer : IOptimizer
{
    private readonly AdaDelta _adaDelta;
    private MemoryBuffer1D<float, Stride1D.Dense> _accumulatedGradients;
    private MemoryBuffer1D<float, Stride1D.Dense> _accumulatedUpdates;

    private Action<Index1D, OptimizerKernelInput, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> _optimizerKernel;

    private OptimizerKernelInput _optimizerKernelInput;

    public AdadeltaOptimizer(AdaDelta adaDelta)
    {
        _adaDelta = adaDelta;
    }

    public void Compile(NetworkTrainer trainer)
    {
        _optimizerKernelInput = new OptimizerKernelInput
        {
            BatchSize = trainer.Buffers.BatchSize,
            ParameterCount = trainer.Network.NetworkData.ParameterCount,
            Rho = _adaDelta.Rho,
            Epsilon = _adaDelta.Epsilon,
            LearningRate = 0.1f
        };
        _accumulatedGradients = trainer.Accelerator.Allocate1D<float>(trainer.Network.NetworkData.ParameterCount);
        _accumulatedUpdates = trainer.Accelerator.Allocate1D<float>(trainer.Network.NetworkData.ParameterCount);
        _optimizerKernel =
            trainer.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, OptimizerKernelInput, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(AdadeltaOptimizerKernelImpl);
    }

    public void Optimize(NetworkData networkData, NetworkAcceleratorBuffers buffers, float learningRate)
    {
        _optimizerKernelInput.LearningRate = learningRate;
        _optimizerKernel(networkData.ParameterCount, _optimizerKernelInput, buffers.Parameters.View,
            buffers.Gradients.View, _accumulatedGradients.View, _accumulatedUpdates.View);
    }

    public void Dispose()
    {
        _accumulatedGradients.Dispose();
        _accumulatedUpdates.Dispose();
    }

    public static void AdadeltaOptimizerKernelImpl(
        Index1D index,
        OptimizerKernelInput input,
        ArrayView<float> parameters,
        ArrayView<float> gradients,
        ArrayView<float> accumulatedGradients,
        ArrayView<float> accumulatedUpdates)
    {
        // Accumulate gradients across the batch for each parameter
        var gradientSum = 0.0f;
        for (var i = 0; i < input.BatchSize; i++) gradientSum += gradients[i * input.ParameterCount + index];

        // Average the gradient over the batch size
        gradientSum /= input.BatchSize;

        // Update accumulated gradients
        accumulatedGradients[index] =
            input.Rho * accumulatedGradients[index] + (1 - input.Rho) * gradientSum * gradientSum;

        // Compute update step
        var updateStep = MathF.Sqrt(accumulatedUpdates[index] + input.Epsilon) /
                         MathF.Sqrt(accumulatedGradients[index] + input.Epsilon);

        // Update accumulated updates
        accumulatedUpdates[index] = input.Rho * accumulatedUpdates[index] + (1 - input.Rho) * gradientSum * gradientSum;

        // Update parameter using Adadelta update rule
        parameters[index] -= input.LearningRate * updateStep;
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