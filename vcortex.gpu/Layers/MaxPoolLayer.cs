using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using vcortex.Core;
using vcortex.Core.Layers;

namespace vcortex.gpu.Layers;

public class MaxPoolLayer : IConvolutionalLayer
{
    public MaxPoolLayer(int poolSize)
    {
        PoolSize = poolSize;
    }

    public int PoolSize { get; }
    
    public int OutputWidth => InputWidth / PoolSize;
    public int OutputHeight => InputHeight / PoolSize;

    public int NumInputs => InputWidth * InputHeight * InputChannels;
    public int NumOutputs => OutputWidth * OutputHeight * OutputChannels;

    public int InputWidth { get; private set; }
    public int InputHeight { get; private set; }
    public int InputChannels { get; private set; }
    public int OutputChannels { get; private set; }
    public int ActivationInputOffset { get; private set; }
    public int ActivationOutputOffset { get; private set; }
    public int CurrentLayerErrorOffset { get; private set; }
    public int NextLayerErrorOffset { get; private set; }
    public int ParameterCount { get; private set; }
    public int ParameterOffset { get; private set; }
    private ForwardKernelInputs _forwardKernelInputs;
    private BackwardKernelInputs _backwardKernelInputs;
    public struct ForwardKernelInputs
    {
        public required  int ActivationInputOffset{ get; set; }
        public required  int ActivationOutputOffset{ get; set; }
        public required  int ParameterOffset{ get; set; }
        public required  int InputWidth{ get; set; }
        public required  int InputHeight{ get; set; }
        public required  int OutputWidth{ get; set; }
        public required  int OutputHeight{ get; set; }
        public required  int InputChannels{ get; set; }
        public required int ActivationCount { get; set; }
        public required int PoolSize { get; set; }
    }
    
    public struct BackwardKernelInputs
    {
        public required int ActivationInputOffset{ get; set; }
        public required int NextLayerErrorOffset{ get; set; }
        public required int CurrentLayerErrorOffset{ get; set; }
        public required int ParameterOffset{ get; set; }
        public required int InputWidth{ get; set; }
        public required int InputHeight{ get; set; }
        public required int OutputWidth{ get; set; }
        public required int OutputHeight{ get; set; }
        public required int InputChannels{ get; set; }
        public required int ActivationCount { get; set; }
        public required int ParameterCount { get; set; }
        public required int PoolSize { get; set; }
    }
    

    public void Connect(IConvolutionalLayer prevLayer)
    {
        InputWidth = prevLayer.OutputWidth;
        InputHeight = prevLayer.OutputHeight;
        OutputChannels = InputChannels = prevLayer.OutputChannels;

        ParameterCount = 0;
        ParameterOffset = prevLayer.ParameterOffset + prevLayer.ParameterCount;

        ActivationInputOffset = prevLayer.ActivationOutputOffset;
        ActivationOutputOffset = prevLayer.ActivationOutputOffset + prevLayer.NumOutputs;
        CurrentLayerErrorOffset = prevLayer.NextLayerErrorOffset;
        NextLayerErrorOffset = prevLayer.CurrentLayerErrorOffset + NumInputs;

        LayerData = new LayerData(NumInputs, NumOutputs, ActivationInputOffset, ActivationOutputOffset,
            NextLayerErrorOffset, CurrentLayerErrorOffset, ParameterOffset, 0, InputWidth, InputHeight, OutputWidth,
            OutputHeight, InputChannels, OutputChannels, 0, 0, PoolSize);
    }

    public void Connect(ConvolutionInputConfig config)
    {
        InputWidth = config.Width;
        InputHeight = config.Height;
        OutputChannels = InputChannels = config.Grayscale ? 1 : 3;

        ParameterCount = 0;
        ParameterOffset = 0;

        ActivationInputOffset = 0;
        ActivationOutputOffset = NumInputs;
        CurrentLayerErrorOffset = 0;
        NextLayerErrorOffset = NumInputs;

        LayerData = new LayerData(NumInputs, NumOutputs, ActivationInputOffset, ActivationOutputOffset,
            NextLayerErrorOffset, CurrentLayerErrorOffset, ParameterOffset, 0, InputWidth, InputHeight, OutputWidth,
            OutputHeight, InputChannels, OutputChannels, 0, 0, PoolSize);
    }


    public LayerData LayerData { get; set; }

    #region Kernels

    private Action<Index1D, ForwardKernelInputs, ArrayView<float>> _forwardKernel;

    private Action<Index1D, BackwardKernelInputs, ArrayView<float>, ArrayView<float>> _backwardKernel;
    public void CompileKernels(INetworkAgent agent)
    {
        _forwardKernelInputs = new ForwardKernelInputs
        {
            ParameterOffset = ParameterOffset,
            ActivationCount = agent.Network.NetworkData.ActivationCount,
            ActivationInputOffset = ActivationInputOffset,
            ActivationOutputOffset = ActivationOutputOffset,
            InputWidth = InputWidth,
            InputHeight = InputHeight,
            OutputWidth = OutputWidth,
            OutputHeight = OutputHeight,
            InputChannels = InputChannels,
            PoolSize = PoolSize,
        };
        _backwardKernelInputs = new BackwardKernelInputs
        {
            ParameterOffset = ParameterOffset,
            ActivationCount = agent.Network.NetworkData.ActivationCount,
            ActivationInputOffset = ActivationInputOffset,
            CurrentLayerErrorOffset = CurrentLayerErrorOffset,
            ParameterCount = agent.Network.NetworkData.ParameterCount,
            NextLayerErrorOffset = NextLayerErrorOffset,
            InputWidth = InputWidth,
            InputHeight = InputHeight,
            OutputWidth = OutputWidth,
            OutputHeight = OutputHeight,
            InputChannels = InputChannels,
            PoolSize = PoolSize,
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
            agent.Buffers.BatchSize * LayerData.InputChannels * LayerData.OutputHeight *
            LayerData.OutputWidth, _forwardKernelInputs, agent.Buffers.Activations.View);
    }

    public void Backward(NetworkTrainer trainer)
    {
        _backwardKernel(
            trainer.Buffers.BatchSize * LayerData.InputChannels * LayerData.OutputHeight *
            LayerData.OutputWidth, _backwardKernelInputs, trainer.Buffers.Activations.View,
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
        if (maxIndex >= 0)
        {
            errors[currentErrorOffset + maxIndex] = errors[nextErrorOffset + outputIndex];
        }
    }
    #endregion

}