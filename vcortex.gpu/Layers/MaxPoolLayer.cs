using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using vcortex.Core.Layers;

namespace vcortex.gpu.Layers;

public class MaxPoolLayer : IConvolutionalLayer
{
    private readonly Maxpool _maxpool;
    private BackwardKernelInputs _backwardKernelInputs;

    private ForwardKernelInputs _forwardKernelInputs;

    public MaxPoolLayer(Maxpool maxpool)
    {
        _maxpool = maxpool;
    }

    public Layer Config => _maxpool;

    public struct ForwardKernelInputs
    {
        public int ActivationInputOffset { get; set; }
        public int ActivationOutputOffset { get; set; }
        public int ParameterOffset { get; set; }
        public int InputWidth { get; set; }
        public int InputHeight { get; set; }
        public int OutputWidth { get; set; }
        public int OutputHeight { get; set; }
        public int InputChannels { get; set; }
        public int ActivationCount { get; set; }
        public int PoolSize { get; set; }
    }

    public struct BackwardKernelInputs
    {
        public int ActivationInputOffset { get; set; }
        public int NextLayerErrorOffset { get; set; }
        public int CurrentLayerErrorOffset { get; set; }
        public int ParameterOffset { get; set; }
        public int InputWidth { get; set; }
        public int InputHeight { get; set; }
        public int OutputWidth { get; set; }
        public int OutputHeight { get; set; }
        public int InputChannels { get; set; }
        public int ActivationCount { get; set; }
        public int ParameterCount { get; set; }
        public int PoolSize { get; set; }
    }

    #region Kernels

    private Action<Index1D, ForwardKernelInputs, ArrayView<float>> _forwardKernel;

    private Action<Index1D, BackwardKernelInputs, ArrayView<float>, ArrayView<float>> _backwardKernel;

    public void CompileKernels(INetworkAgent agent)
    {
        _forwardKernelInputs = new ForwardKernelInputs
        {
            ParameterOffset = _maxpool.ParameterOffset,
            ActivationCount = agent.Network.NetworkData.ActivationCount,
            ActivationInputOffset = _maxpool.ActivationInputOffset,
            ActivationOutputOffset = _maxpool.ActivationOutputOffset,
            InputWidth = _maxpool.InputWidth,
            InputHeight = _maxpool.InputHeight,
            OutputWidth = _maxpool.OutputWidth,
            OutputHeight = _maxpool.OutputHeight,
            InputChannels = _maxpool.InputChannels,
            PoolSize = _maxpool.PoolSize
        };
        _backwardKernelInputs = new BackwardKernelInputs
        {
            ParameterOffset = _maxpool.ParameterOffset,
            ActivationCount = agent.Network.NetworkData.ActivationCount,
            ActivationInputOffset = _maxpool.ActivationInputOffset,
            CurrentLayerErrorOffset = _maxpool.CurrentLayerErrorOffset,
            ParameterCount = agent.Network.NetworkData.ParameterCount,
            NextLayerErrorOffset = _maxpool.NextLayerErrorOffset,
            InputWidth = _maxpool.InputWidth,
            InputHeight = _maxpool.InputHeight,
            OutputWidth = _maxpool.OutputWidth,
            OutputHeight = _maxpool.OutputHeight,
            InputChannels = _maxpool.InputChannels,
            PoolSize = _maxpool.PoolSize
        };

        _forwardKernel =
            agent.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ForwardKernelInputs, ArrayView<float>>(
                ForwardKernel);
        _backwardKernel =
            agent.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, BackwardKernelInputs, ArrayView<float>, ArrayView<float>>(
                    BackwardKernel);
    }

    public void FillRandom(INetworkAgent agent)
    {
    }

    public void Forward(INetworkAgent agent)
    {
        _forwardKernel(
            agent.Buffers.BatchSize * _maxpool.InputChannels * _maxpool.OutputHeight *
            _maxpool.OutputWidth, _forwardKernelInputs, agent.Buffers.Activations.View);
    }

    public void Backward(NetworkTrainer trainer)
    {
        _backwardKernel(
            trainer.Buffers.BatchSize * _maxpool.InputChannels * _maxpool.OutputHeight *
            _maxpool.OutputWidth, _backwardKernelInputs, trainer.Buffers.Activations.View,
            trainer.Buffers.Errors.View);
    }

    public static void ForwardKernel(
        Index1D index,
        ForwardKernelInputs inputs,
        ArrayView<float> activations)
    {
        // Calculate indices for batch, channel, output y, and output x
        int batch = index / (inputs.InputChannels * inputs.OutputHeight * inputs.OutputWidth);
        var c = index / (inputs.OutputHeight * inputs.OutputWidth) % inputs.InputChannels;
        var y = index / inputs.OutputWidth % inputs.OutputHeight;
        var x = index % inputs.OutputWidth;

        // Offsets to access input and output activations for this batch
        var activationInputOffset = batch * inputs.ActivationCount + inputs.ActivationInputOffset;
        var activationOutputOffset = batch * inputs.ActivationCount + inputs.ActivationOutputOffset;

        // Offsets for channel positions in input and output layers
        var inputChannelOffset = inputs.InputWidth * inputs.InputHeight;
        var outputChannelOffset = inputs.OutputWidth * inputs.OutputHeight;

        // Calculate output index for storing the maximum pooled value
        var outputIndex = y * inputs.OutputWidth + x + c * outputChannelOffset;

        // Initialize max value to minimum possible float
        var max = float.MinValue;

        // Traverse the pooling window and find the maximum activation
        for (var ky = 0; ky < inputs.PoolSize; ky++)
        for (var kx = 0; kx < inputs.PoolSize; kx++)
        {
            // Map pooling coordinates (kx, ky) to input coordinates
            var oldX = x * inputs.PoolSize + kx;
            var oldY = y * inputs.PoolSize + ky;

            // Calculate the input index for the current (oldY, oldX) position
            var inputIndex = oldY * inputs.InputWidth + oldX + c * inputChannelOffset;

            // Update max if the current input activation is greater
            max = XMath.Max(max, activations[activationInputOffset + inputIndex]);
        }

        // Store the maximum value in the output activation array
        activations[activationOutputOffset + outputIndex] = max;
    }

    public static void BackwardKernel(
        Index1D index,
        BackwardKernelInputs inputs,
        ArrayView<float> activations,
        ArrayView<float> errors)
    {
        // Calculate batch, channel, output y, and output x
        int batch = index / (inputs.InputChannels * inputs.OutputHeight * inputs.OutputWidth);
        var c = index / (inputs.OutputHeight * inputs.OutputWidth) % inputs.InputChannels;
        var y = index / inputs.OutputWidth % inputs.OutputHeight;
        var x = index % inputs.OutputWidth;

        // Offsets for activations and errors in the current and next layers
        var activationInputOffset = batch * inputs.ActivationCount + inputs.ActivationInputOffset;
        var currentErrorOffset = batch * inputs.ActivationCount + inputs.CurrentLayerErrorOffset;
        var nextErrorOffset = batch * inputs.ActivationCount + inputs.NextLayerErrorOffset;

        // Offsets for input and output channels
        var inputChannelOffset = inputs.InputWidth * inputs.InputHeight;
        var outputChannelOffset = inputs.OutputWidth * inputs.OutputHeight;

        // Index in the output error for the current pooling position
        var outputIndex = y * inputs.OutputWidth + x + c * outputChannelOffset;

        // Initialize max value to track the maximum and its position
        var max = float.MinValue;
        var maxIndex = 0;

        // Traverse the pooling window to identify the maximum activation and its index
        for (var ky = 0; ky < inputs.PoolSize; ky++)
        for (var kx = 0; kx < inputs.PoolSize; kx++)
        {
            // Calculate input coordinates (oldX, oldY) for the pooling window
            var oldX = x * inputs.PoolSize + kx;
            var oldY = y * inputs.PoolSize + ky;

            // Compute input index for this (oldY, oldX) within the channel
            var inputIndex = oldY * inputs.InputWidth + oldX + c * inputChannelOffset;

            // Update the maximum and record the index if a new max is found
            if (activations[activationInputOffset + inputIndex] >= max)
            {
                max = activations[activationInputOffset + inputIndex];
                maxIndex = inputIndex;
            }
        }

        // propagate the error to the position where the max was found
        if (maxIndex >= 0) errors[currentErrorOffset + maxIndex] = errors[nextErrorOffset + outputIndex];
    }

    #endregion
}