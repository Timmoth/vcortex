using ILGPU;
using ILGPU.Runtime;
using vcortex.Core;
using vcortex.Core.Optimizers;

namespace vcortex.gpu.Optimizers;

public class SGDWithMomentumOptimizer : IOptimizer
{
    private MemoryBuffer1D<float, Stride1D.Dense> _velocity;
    private Action<Index1D, OptimizerKernelInput, ArrayView<float>, ArrayView<float>, ArrayView<float>> _optimizerKernel;

    private readonly SgdMomentum _sgdMomentum;
    public struct OptimizerKernelInput
    {
        public int BatchSize { get; set; }
        public int ParameterCount { get; set; }
        public float Momentum { get; set; }
        public float LearningRate { get; set; }
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
    private OptimizerKernelInput _optimizerKernelInput;

    public SGDWithMomentumOptimizer(SgdMomentum sgdMomentum)
    {
        _sgdMomentum = sgdMomentum;
    }

    public void Compile(NetworkTrainer trainer)
    {
        _optimizerKernelInput = new OptimizerKernelInput()
        {
            BatchSize = trainer.Buffers.BatchSize,
            LearningRate = 0.01f,
            Momentum = _sgdMomentum.Momentum,
            ParameterCount = trainer.Network.ParameterCount,
        };
        _velocity = trainer.Accelerator.Allocate1D<float>(trainer.Network.ParameterCount);
        _optimizerKernel =
            trainer.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, OptimizerKernelInput, ArrayView<float>, ArrayView<float>, ArrayView<float>>(SGDWithMomentumOptimizerKernelImpl);
    }

    public void Optimize(NetworkData networkData, NetworkAcceleratorBuffers buffers, float learningRate)
    {
        _optimizerKernelInput.LearningRate = learningRate;
        _optimizerKernel(networkData.ParameterCount, _optimizerKernelInput, buffers.Parameters.View, buffers.Gradients.View, _velocity.View);
    }

    public void Dispose() { _velocity.Dispose(); }
}