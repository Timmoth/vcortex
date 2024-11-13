using ILGPU;
using ILGPU.Runtime;
using vcortex.Optimizers;

namespace vcortex.gpu.Optimizers;

public class SGDWithMomentumOptimizer : IOptimizer
{
    private readonly SgdMomentum _sgdMomentum;

    private readonly Action<Index1D, OptimizerKernelInput, ArrayView<float>, ArrayView<float>, ArrayView<float>>
        _optimizerKernel;

    private OptimizerKernelInput _optimizerKernelInput;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _velocity;
    private readonly NetworkTrainer _trainer;

    public SGDWithMomentumOptimizer(SgdMomentum sgdMomentum, NetworkTrainer trainer)
    {
        _sgdMomentum = sgdMomentum;
        _trainer = trainer;

        _optimizerKernelInput = new OptimizerKernelInput
        {
            BatchSize = trainer.Buffers.BatchSize,
            LearningRate = 0.01f,
            Momentum = _sgdMomentum.Momentum,
            ParameterCount = trainer.Network.NetworkData.ParameterCount
        };
        _velocity = trainer.Accelerator.Allocate1D<float>(trainer.Network.NetworkData.ParameterCount);
        _optimizerKernel =
            trainer.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, OptimizerKernelInput, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>>(SGDWithMomentumOptimizerKernelImpl);
    }

    public void Optimize(float learningRate)
    {
        _optimizerKernelInput.LearningRate = learningRate;
        _optimizerKernel(_trainer.Network.NetworkData.ParameterCount, _optimizerKernelInput, _trainer.Buffers.Parameters.View,
            _trainer.Buffers.Gradients.View, _velocity.View);
    }

    public void Dispose()
    {
        _velocity.Dispose();
    }

    public static void SGDWithMomentumOptimizerKernelImpl(
        Index1D index,
        OptimizerKernelInput input,
        ArrayView<float> parameters,
        ArrayView<float> gradients,
        ArrayView<float> velocity)
    {
        var batchSize = input.BatchSize;

        // Accumulate gradients across the batch for each parameter
        var gradientSum = 0.0f;
        for (var i = 0; i < batchSize; i++) gradientSum += gradients[i * input.ParameterCount + index];

        // Average the gradient over the batch size
        gradientSum /= batchSize;

        // Update the momentum term
        velocity[index] = input.Momentum * velocity[index] + (1 - input.Momentum) * gradientSum;

        // Update the parameter using SGD with momentum update rule
        parameters[index] -= input.LearningRate * velocity[index];
    }

    public struct OptimizerKernelInput
    {
        public int BatchSize { get; set; }
        public int ParameterCount { get; set; }
        public float Momentum { get; set; }
        public float LearningRate { get; set; }
    }
}