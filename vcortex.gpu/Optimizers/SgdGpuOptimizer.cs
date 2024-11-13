using ILGPU;
using ILGPU.Runtime;
using vcortex.Core;
using vcortex.Core.Optimizers;

namespace vcortex.gpu.Optimizers;

public class SgdOptimizer : IOptimizer
{
    private Action<Index1D, OptimizerKernelInput, ArrayView<float>, ArrayView<float>> _optimizerKernel;

    private readonly Sgd _sgd;
    public struct OptimizerKernelInput
    {
        public int BatchSize { get; set; }
        public int ParameterCount { get; set; }
        public float LearningRate { get; set; }
    }
    
    public static void SGDOptimizerKernelImpl(
        Index1D index,
        OptimizerKernelInput input,
        ArrayView<float> parameters,
        ArrayView<float> gradients)
    {
        var batchSize = input.BatchSize;
    
        // Accumulate gradients across the batch for each parameter
        var gradientSum = 0.0f;
        for (var i = 0; i < batchSize; i++) gradientSum += gradients[i * input.ParameterCount + index];

        // Average the gradient over the batch size
        gradientSum /= batchSize;

        // Update parameter using SGD update rule
        parameters[index] -= input.LearningRate * gradientSum;
    }
    
    private OptimizerKernelInput _optimizerKernelInput;

    public SgdOptimizer(Sgd sgd)
    {
        _sgd = sgd;
    }

    public void Compile(NetworkTrainer trainer)
    {
        _optimizerKernelInput = new OptimizerKernelInput()
        {
            BatchSize = trainer.Buffers.BatchSize,
            ParameterCount = trainer.Network.ParameterCount,
            LearningRate = 0.01f
        };
        _optimizerKernel =
            trainer.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, OptimizerKernelInput, ArrayView<float>, ArrayView<float>>(SGDOptimizerKernelImpl);
        
    }

    public void Optimize(NetworkData networkData, NetworkAcceleratorBuffers buffers, float learningRate)
    {
        _optimizerKernelInput.LearningRate = learningRate;
        _optimizerKernel(networkData.ParameterCount, _optimizerKernelInput, buffers.Parameters.View, buffers.Gradients.View);
    }

    public void Dispose()
    {
    }
}