using ILGPU;
using ILGPU.Runtime;
using vcortex.Network;
using vcortex.Optimizers;

namespace vcortex.gpu.Optimizers;

public class AdamOptimizer : IOptimizer
{
    private readonly Adam _adam;

    private readonly MemoryBuffer1D<float, Stride1D.Dense> _firstMoment;

    private readonly Action<Index1D, OptimizerKernelInput, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> _optimizerKernel;

    private OptimizerKernelInput _optimizerKernelInput;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _secondMoment;
    public int Timestep;
    private readonly GpuNetworkTrainer _trainer;

    public AdamOptimizer(Adam adam, GpuNetworkTrainer trainer)
    {
        _adam = adam;
        _trainer = trainer;

        _optimizerKernelInput = new OptimizerKernelInput
        {
            BatchSize = trainer.Buffers.BatchSize,
            ParameterCount = trainer.Network.NetworkData.ParameterCount,
            Beta1 = _adam.Beta1,
            Beta2 = _adam.Beta2,
            Epsilon = _adam.Epsilon,
            BiasCorrection1 = 1 - MathF.Pow(0.9f, Timestep),
            BiasCorrection2 = 1 - MathF.Pow(0.999f, Timestep)
        };
        _firstMoment = trainer.Accelerator.Allocate1D<float>(trainer.Network.NetworkData.ParameterCount);
        _secondMoment = trainer.Accelerator.Allocate1D<float>(trainer.Network.NetworkData.ParameterCount);
        _optimizerKernel =
            trainer.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, OptimizerKernelInput, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(AdamOptimizerKernelImpl);
    }

    public void Optimize(float learningRate)
    {
        Timestep++;
        _optimizerKernelInput.LearningRate = learningRate;
        _optimizerKernelInput.BiasCorrection1 = 1 - MathF.Pow(_optimizerKernelInput.Beta1, Timestep);
        _optimizerKernelInput.BiasCorrection2 = 1 - MathF.Pow(_optimizerKernelInput.Beta2, Timestep);

        _optimizerKernel(_trainer.Network.NetworkData.ParameterCount, _optimizerKernelInput, _trainer.Buffers.Parameters.View,
            _trainer.Buffers.Gradients.View, _firstMoment.View, _secondMoment.View);
    }

    public void Reset()
    {
        Timestep = 0;
    }

    public void Dispose()
    {
        _firstMoment.Dispose();
        _secondMoment.Dispose();
    }

    public static void AdamOptimizerKernelImpl(
        Index1D index,
        OptimizerKernelInput input,
        ArrayView<float> parameters,
        ArrayView<float> gradients,
        ArrayView<float> firstMoment,
        ArrayView<float> secondMoment)
    {
        var batchSize = input.BatchSize;

        // Accumulate gradients across the batch for each parameter
        var gradientSum = 0.0f;
        for (var i = 0; i < batchSize; i++) gradientSum += gradients[i * input.ParameterCount + index];

        // Average the gradient over the batch size
        gradientSum /= batchSize;

        // Update the first and second moment estimates for the Adam optimizer
        firstMoment[index] =
            input.Beta1 * firstMoment[index] + (1 - input.Beta1) * gradientSum;
        secondMoment[index] = input.Beta2 * secondMoment[index] +
                              (1 - input.Beta2) * gradientSum * gradientSum;

        // Apply bias correction for the moments
        var mHat = firstMoment[index] / input.BiasCorrection1;
        var vHat = secondMoment[index] / input.BiasCorrection2;

        // Update parameter using Adam optimizer formula
        parameters[index] -= input.LearningRate * mHat / (MathF.Sqrt(vHat) + input.Epsilon);
    }

    public struct OptimizerKernelInput
    {
        public int BatchSize { get; set; }
        public int ParameterCount { get; set; }
        public float Beta1 { get; set; }
        public float Beta2 { get; set; }
        public float BiasCorrection1 { get; set; }
        public float BiasCorrection2 { get; set; }
        public float Epsilon { get; set; }
        public float LearningRate { get; set; }
    }
}