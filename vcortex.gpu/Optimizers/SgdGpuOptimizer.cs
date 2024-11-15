using ILGPU;
using ILGPU.Runtime;
using vcortex.Network;
using vcortex.Optimizers;

namespace vcortex.gpu.Optimizers;

public class SgdOptimizer : IOptimizer
{
    private readonly Sgd _sgd;
    private readonly Action<Index1D, OptimizerKernelInput, ArrayView<float>, ArrayView<float>> _optimizerKernel;

    private OptimizerKernelInput _optimizerKernelInput;
    private readonly GpuNetworkTrainer _trainer;

    public SgdOptimizer(Sgd sgd, GpuNetworkTrainer trainer)
    {
        _sgd = sgd;
        _trainer = trainer;

        _optimizerKernelInput = new OptimizerKernelInput
        {
            BatchSize = trainer.Buffers.BatchSize,
            ParameterCount = trainer.Network.NetworkData.ParameterCount,
            LearningRate = 0.01f
        };
        _optimizerKernel =
            trainer.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, OptimizerKernelInput, ArrayView<float>, ArrayView<float>>(
                    SGDOptimizerKernelImpl);
    }

    public void Optimize(float learningRate)
    {
        _optimizerKernelInput.LearningRate = learningRate;
        _optimizerKernel(_trainer.Network.NetworkData.ParameterCount, _optimizerKernelInput, _trainer.Buffers.Parameters.View,
            _trainer.Buffers.Gradients.View);
    }

    public void Dispose()
    {
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

    public struct OptimizerKernelInput
    {
        public int BatchSize { get; set; }
        public int ParameterCount { get; set; }
        public float LearningRate { get; set; }
    }
}