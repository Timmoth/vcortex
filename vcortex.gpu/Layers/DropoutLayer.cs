using ILGPU;
using ILGPU.Algorithms.Random;
using ILGPU.Runtime;
using vcortex.Layers;
using vcortex.Network;

namespace vcortex.gpu.Layers;

public class DropoutLayer : IConnectedLayer
{
    private readonly Dropout _dropout;
    private readonly NetworkAcceleratorBuffers _buffers;
    private readonly Accelerator _accelerator;
    private BackwardKernelInputs _backwardKernelInputs;

    private ForwardKernelInputs _forwardKernelInputs;
    public bool IsTraining { get; set; }
    public DropoutLayer(Dropout dropout, NetworkAcceleratorBuffers buffers, Accelerator accelerator, NetworkData networkData)
    {
        _dropout = dropout;
        _buffers = buffers;
        _accelerator = accelerator;

        _forwardKernelInputs = new ForwardKernelInputs
        {
            ActivationCount = networkData.ActivationCount,
            ActivationInputOffset = _dropout.ActivationInputOffset,
            NumOutputs = _dropout.NumOutputs,
            ActivationOutputOffset = _dropout.ActivationOutputOffset,
            DropRate = _dropout.DropoutRate
        };
        _backwardKernelInputs = new BackwardKernelInputs
        {
            ActivationCount = networkData.ActivationCount,
            NumOutputs = _dropout.NumOutputs,
            CurrentLayerErrorOffset = _dropout.CurrentLayerErrorOffset,
            NextLayerErrorOffset = _dropout.NextLayerErrorOffset,
            DropRate = _dropout.DropoutRate
        };

        Mask = accelerator.Allocate1D<float>(_dropout.NumOutputs);

        var random = new Random();
        Rng = RNG.Create<XorShift64Star>(accelerator, random);

        _forwardKernel =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, ForwardKernelInputs, ArrayView<float>, ArrayView<float>>(
                    ForwardKernel);
        _backwardKernel =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, BackwardKernelInputs, ArrayView<float>, ArrayView<float>>(
                    BackwardKernel);
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

    private readonly Action<Index1D, ForwardKernelInputs, ArrayView<float>, ArrayView<float>> _forwardKernel;
    private readonly Action<Index1D, BackwardKernelInputs, ArrayView<float>, ArrayView<float>> _backwardKernel;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> Mask;
    private readonly RNG<XorShift64Star> Rng;


    public void FillRandom()
    {
    }

    public void Forward()
    {
        Rng.FillUniform(_accelerator.DefaultStream, Mask.View);

        _forwardKernelInputs.DropRate = IsTraining ? _dropout.DropoutRate : 0.0f;
        _forwardKernel(_buffers.BatchSize * _dropout.NumOutputs, _forwardKernelInputs,
            _buffers.Activations.View, Mask.View);
    }

    public void Backward()
    {
        _backwardKernelInputs.DropRate = IsTraining ? _dropout.DropoutRate : 0.0f;
        _backwardKernel(_buffers.BatchSize * _dropout.NumOutputs,
            _backwardKernelInputs, _buffers.Errors.View, Mask.View);
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