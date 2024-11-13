using ILGPU;
using ILGPU.Algorithms.Random;
using ILGPU.Runtime;
using vcortex.Core.Layers;

namespace vcortex.gpu.Layers;

public class DropoutLayer : IConnectedLayer
{
    private readonly Dropout _dropout;
    private BackwardKernelInputs _backwardKernelInputs;

    private ForwardKernelInputs _forwardKernelInputs;

    public DropoutLayer(Dropout dropout)
    {
        _dropout = dropout;
    }

    public Layer Config => _dropout;

    public struct ForwardKernelInputs
    {
        public int ActivationCount { get; set; }
        public int ActivationInputOffset { get; set; }
        public int NumOutputs { get; set; }
        public int ActivationOutputOffset { get; set; }
        public float DropRate { get; set; }
    }

    public struct BackwardKernelInputs
    {
        public int ActivationCount { get; set; }
        public int NumOutputs { get; set; }
        public int CurrentLayerErrorOffset { get; set; }
        public int NextLayerErrorOffset { get; set; }
        public float DropRate { get; set; }
    }

    #region Kernels

    private Action<Index1D, ForwardKernelInputs, ArrayView<float>, ArrayView<float>> _forwardKernel;
    private Action<Index1D, BackwardKernelInputs, ArrayView<float>, ArrayView<float>> _backwardKernel;
    private MemoryBuffer1D<float, Stride1D.Dense> Mask;
    private RNG<XorShift64Star> Rng;


    public void FillRandom(INetworkAgent agent)
    {
    }

    public void Forward(INetworkAgent agent)
    {
        Rng.FillUniform(agent.Accelerator.DefaultStream, Mask.View);

        _forwardKernelInputs.DropRate = agent.IsTraining ? _dropout.DropoutRate : 0.0f;
        _forwardKernel(agent.Buffers.BatchSize * _dropout.NumOutputs, _forwardKernelInputs,
            agent.Buffers.Activations.View, Mask.View);
    }

    public void Backward(NetworkTrainer trainer)
    {
        _backwardKernelInputs.DropRate = trainer.IsTraining ? _dropout.DropoutRate : 0.0f;
        _backwardKernel(trainer.Buffers.BatchSize * _dropout.NumOutputs,
            _backwardKernelInputs, trainer.Buffers.Errors.View, Mask.View);
    }

    public void CompileKernels(INetworkAgent agent)
    {
        _forwardKernelInputs = new ForwardKernelInputs
        {
            ActivationCount = agent.Network.NetworkData.ActivationCount,
            ActivationInputOffset = _dropout.ActivationInputOffset,
            NumOutputs = _dropout.NumOutputs,
            ActivationOutputOffset = _dropout.ActivationOutputOffset,
            DropRate = _dropout.DropoutRate
        };
        _backwardKernelInputs = new BackwardKernelInputs
        {
            ActivationCount = agent.Network.NetworkData.ActivationCount,
            NumOutputs = _dropout.NumOutputs,
            CurrentLayerErrorOffset = _dropout.CurrentLayerErrorOffset,
            NextLayerErrorOffset = _dropout.NextLayerErrorOffset,
            DropRate = _dropout.DropoutRate
        };

        Mask = agent.Accelerator.Allocate1D<float>(_dropout.NumOutputs);

        var random = new Random();
        Rng = RNG.Create<XorShift64Star>(agent.Accelerator, random);

        _forwardKernel =
            agent.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, ForwardKernelInputs, ArrayView<float>, ArrayView<float>>(
                    ForwardKernel);
        _backwardKernel =
            agent.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, BackwardKernelInputs, ArrayView<float>, ArrayView<float>>(
                    BackwardKernel);
    }

    public static void ForwardKernel(
        Index1D index,
        ForwardKernelInputs inputs,
        ArrayView<float> activations,
        ArrayView<float> mask)
    {
        // index = batches * outputs
        int batch = index / inputs.NumOutputs;
        var outputIndex = index % inputs.NumOutputs;

        var activationInputOffset = batch * inputs.ActivationCount + inputs.ActivationInputOffset;
        var activationOutputOffset = batch * inputs.ActivationCount + inputs.ActivationOutputOffset;

        // Apply mask: if neuron is kept, use the input; otherwise, set output to zero
        activations[activationOutputOffset + outputIndex] = mask[outputIndex] >= inputs.DropRate
            ? activations[activationInputOffset + outputIndex]
            : 0.0f;
    }

    public static void BackwardKernel(
        Index1D index,
        BackwardKernelInputs inputs,
        ArrayView<float> errors,
        ArrayView<float> mask)
    {
        int batch = index / inputs.NumOutputs;
        var outputIndex = index % inputs.NumOutputs;

        var currentErrorOffset = batch * inputs.ActivationCount + inputs.CurrentLayerErrorOffset;
        var nextErrorOffset = batch * inputs.ActivationCount + inputs.NextLayerErrorOffset;

        // Only propagate errors for active neurons (those that were not dropped)
        errors[currentErrorOffset + outputIndex] =
            mask[outputIndex] >= inputs.DropRate ? errors[nextErrorOffset + outputIndex] : 0.0f;
    }

    #endregion
}