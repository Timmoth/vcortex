using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using vcortex.Accelerated;

namespace vcortex.Layers.Convolution;

public class MaxPoolingConvolutionLayer : IConvolutionalLayer
{
    public MaxPoolingConvolutionLayer(int poolSize)
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
    public int GradientOffset { get; private set; }
    public int ParameterCount { get; private set; }
    public int ParameterOffset { get; private set; }
    public int GradientCount => 0;

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
        GradientOffset = prevLayer.GradientOffset + prevLayer.GradientCount;


        LayerData = new LayerData(NumInputs, NumOutputs, ActivationInputOffset, ActivationOutputOffset, GradientOffset,
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
        GradientOffset = 0;


        LayerData = new LayerData(NumInputs, NumOutputs, ActivationInputOffset, ActivationOutputOffset, GradientOffset,
            NextLayerErrorOffset, CurrentLayerErrorOffset, ParameterOffset, 0, InputWidth, InputHeight, OutputWidth,
            OutputHeight, InputChannels, OutputChannels, 0, 0, PoolSize);
    }


    public LayerData LayerData { get; set; }

    #region Kernels
    public Action<Index1D, NetworkData, LayerData, ArrayView<float>> _forwardKernel { get; private set; }

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>,
        ArrayView<float>> _backwardKernel { get; private set; }
    public void CompileKernels(NetworkAccelerator accelerator)
    {
        _forwardKernel =
            accelerator.accelerator.LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>>(
                ForwardKernel);
        _backwardKernel =
            accelerator.accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>>(
                    BackwardKernel);
    }

    public void Forward(NetworkAccelerator accelerator)
    {
        _forwardKernel(
            accelerator.Network.NetworkData.BatchSize * LayerData.InputChannels * LayerData.OutputHeight *
            LayerData.OutputWidth, accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Activations.View);
    }

    public void Backward(NetworkAccelerator accelerator)
    {
        _backwardKernel(
            accelerator.Network.NetworkData.BatchSize * LayerData.InputChannels * LayerData.OutputHeight *
            LayerData.OutputWidth, accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Activations.View,
            accelerator.Buffers.Errors.View);
    }
    public void AccumulateGradients(NetworkAccelerator accelerator)
    {
    }

    public static void ForwardKernel(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> activations)
    {
        // Calculate indices for batch, channel, output y, and output x
        int batch = index / (layerData.InputChannels * layerData.OutputHeight * layerData.OutputWidth);
        var c = index / (layerData.OutputHeight * layerData.OutputWidth) % layerData.InputChannels;
        var y = index / layerData.OutputWidth % layerData.OutputHeight;
        var x = index % layerData.OutputWidth;

        // Offsets to access input and output activations for this batch
        var activationInputOffset = batch * networkData.ActivationCount + layerData.ActivationInputOffset;
        var activationOutputOffset = batch * networkData.ActivationCount + layerData.ActivationOutputOffset;

        // Offsets for channel positions in input and output layers
        var inputChannelOffset = layerData.InputWidth * layerData.InputHeight;
        var outputChannelOffset = layerData.OutputWidth * layerData.OutputHeight;

        // Calculate output index for storing the maximum pooled value
        var outputIndex = y * layerData.OutputWidth + x + c * outputChannelOffset;

        // Initialize max value to minimum possible float
        var max = float.MinValue;

        // Traverse the pooling window and find the maximum activation
        for (var ky = 0; ky < layerData.PoolSize; ky++)
        for (var kx = 0; kx < layerData.PoolSize; kx++)
        {
            // Map pooling coordinates (kx, ky) to input coordinates
            var oldX = x * layerData.PoolSize + kx;
            var oldY = y * layerData.PoolSize + ky;

            // Calculate the input index for the current (oldY, oldX) position
            var inputIndex = oldY * layerData.InputWidth + oldX + c * inputChannelOffset;

            // Update max if the current input activation is greater
            max = XMath.Max(max, activations[activationInputOffset + inputIndex]);
        }

        // Store the maximum value in the output activation array
        activations[activationOutputOffset + outputIndex] = max;
    }

    public static void BackwardKernel(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> activations,
        ArrayView<float> errors)
    {
        // Calculate batch, channel, output y, and output x
        int batch = index / (layerData.InputChannels * layerData.OutputHeight * layerData.OutputWidth);
        var c = index / (layerData.OutputHeight * layerData.OutputWidth) % layerData.InputChannels;
        var y = index / layerData.OutputWidth % layerData.OutputHeight;
        var x = index % layerData.OutputWidth;

        // Offsets for activations and errors in the current and next layers
        var activationInputOffset = batch * networkData.ActivationCount + layerData.ActivationInputOffset;
        var currentErrorOffset = batch * networkData.ErrorCount + layerData.CurrentLayerErrorOffset;
        var nextErrorOffset = batch * networkData.ErrorCount + layerData.NextLayerErrorOffset;

        // Offsets for input and output channels
        var inputChannelOffset = layerData.InputWidth * layerData.InputHeight;
        var outputChannelOffset = layerData.OutputWidth * layerData.OutputHeight;

        // Index in the output error for the current pooling position
        var outputIndex = y * layerData.OutputWidth + x + c * outputChannelOffset;

        // Initialize max value to track the maximum and its position
        var max = float.MinValue;
        var maxIndex = 0;

        // Traverse the pooling window to identify the maximum activation and its index
        for (var ky = 0; ky < layerData.PoolSize; ky++)
        for (var kx = 0; kx < layerData.PoolSize; kx++)
        {
            // Calculate input coordinates (oldX, oldY) for the pooling window
            var oldX = x * layerData.PoolSize + kx;
            var oldY = y * layerData.PoolSize + ky;

            // Compute input index for this (oldY, oldX) within the channel
            var inputIndex = oldY * layerData.InputWidth + oldX + c * inputChannelOffset;

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