using ILGPU.Runtime;
using ILGPU;
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
    public int NumOutputs => OutputWidth * OutputHeight * InputChannels;

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
    public float[] Parameters { get; set; }

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

        LayerData = new LayerData()
        {
            ActivationInputOffset = ActivationInputOffset,
            ActivationOutputOffset = ActivationOutputOffset,
            CurrentLayerErrorOffset = CurrentLayerErrorOffset,
            NextLayerErrorOffset = NextLayerErrorOffset,
            GradientOffset = GradientOffset,
            NumInputs = NumInputs,
            NumOutputs = NumOutputs,
            InputWidth = InputWidth,
            InputHeight = InputHeight,
            OutputWidth = OutputWidth,
            OutputHeight = OutputHeight,
            InputChannels = InputChannels,
            OutputChannels = OutputChannels,
            PoolSize = PoolSize
        };
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

        LayerData = new LayerData()
        {
            ActivationInputOffset = ActivationInputOffset,
            ActivationOutputOffset = ActivationOutputOffset,
            CurrentLayerErrorOffset = CurrentLayerErrorOffset,
            NextLayerErrorOffset = NextLayerErrorOffset,
            GradientOffset = GradientOffset,
            NumInputs = NumInputs,
            NumOutputs = NumOutputs,
            InputWidth = InputWidth,
            InputHeight = InputHeight,
            OutputWidth = OutputWidth,
            OutputHeight = OutputHeight,
            InputChannels = InputChannels,
            OutputChannels = OutputChannels,
            PoolSize = PoolSize
        };
    }

    public void Forward(float[] activations)
    {
        var inputChannelOffset = InputWidth * InputHeight;
        var outputChannelOffset = OutputWidth * OutputHeight;

        for (var c = 0; c < InputChannels; c++)
            for (var y = 0; y < OutputHeight; y++)
                for (var x = 0; x < OutputWidth; x++)
                {
                    var outputIndex = y * OutputWidth + x + c * outputChannelOffset;

                    var max = float.MinValue;

                    // SIMD for pooling region, process multiple elements at once
                    for (var ky = 0; ky < PoolSize; ky++)
                        for (var kx = 0; kx < PoolSize; kx++)
                        {
                            var oldX = x * PoolSize + kx;
                            var oldY = y * PoolSize + ky;
                            var inputIndex = oldY * InputWidth + oldX + c * inputChannelOffset;

                            // Here, SIMD could help if PoolSize is large, by comparing multiple input elements at once
                            max = Math.Max(max, activations[ActivationInputOffset + inputIndex]);
                        }

                    activations[ActivationOutputOffset + outputIndex] = max;
                }
    }

    public int GradientCount => 0;

    public void Backward(float[] activations, float[] errors,
        float[] gradients, float learningRate)
    {
        Array.Clear(errors, CurrentLayerErrorOffset, NumInputs);
        var inputChannelOffset = InputWidth * InputHeight;
        var outputChannelOffset = OutputWidth * OutputHeight;

        for (var c = 0; c < InputChannels; c++)
            for (var y = 0; y < OutputHeight; y++)
                for (var x = 0; x < OutputWidth; x++)
                {
                    var outputIndex = y * OutputWidth + x + c * outputChannelOffset;

                    var max = float.MinValue;
                    var maxIndex = -1;

                    // Find max position within pool window
                    for (var ky = 0; ky < PoolSize; ky++)
                        for (var kx = 0; kx < PoolSize; kx++)
                        {
                            var oldX = x * PoolSize + kx;
                            var oldY = y * PoolSize + ky;
                            var inputIndex = oldY * InputWidth + oldX + c * inputChannelOffset;

                            if (activations[ActivationInputOffset + inputIndex] > max)
                            {
                                max = activations[ActivationInputOffset + inputIndex];
                                maxIndex = inputIndex;
                            }
                        }

                    if (maxIndex >= 0)
                        // Propagate error to the max position only
                        errors[CurrentLayerErrorOffset + maxIndex] = errors[NextLayerErrorOffset + outputIndex];
                }
    }

    public void AccumulateGradients(float[][] gradients, float learningRate)
    {
    }

    public void Forward(NetworkAccelerator accelerator)
    {
        ForwardKernel(accelerator.Network.NetworkData.BatchSize * LayerData.InputChannels * LayerData.OutputHeight * LayerData.OutputWidth, accelerator.Network.NetworkData, LayerData,accelerator.Buffers.Activations.View);
    }

    public void Backward(NetworkAccelerator accelerator)
    {
        BackwardKernel(accelerator.Network.NetworkData.BatchSize * LayerData.InputChannels * LayerData.OutputHeight * LayerData.OutputWidth, accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Activations.View, accelerator.Buffers.Errors.View);
    }

    public static void ForwardKernelImpl(
     Index1D index,
     NetworkData networkData,
     LayerData layerData,
     ArrayView<float> activations)
    {
        // index = batches * InputChannels * height * width
        int batch = index / (layerData.InputChannels * layerData.OutputHeight * layerData.OutputWidth);
        int c = (index / (layerData.OutputHeight * layerData.OutputWidth)) % layerData.InputChannels;
        int y = (index / layerData.OutputWidth) % layerData.OutputHeight;
        int x = index % layerData.OutputWidth;

        var activationInputOffset = batch * networkData.ActivationCount + layerData.ActivationInputOffset;
        var activationOutputOffset = batch * networkData.ActivationCount + layerData.ActivationOutputOffset;

        var inputChannelOffset = layerData.InputWidth * layerData.InputHeight;
        var outputChannelOffset = layerData.OutputWidth * layerData.OutputHeight;

        var outputIndex = y * layerData.OutputWidth + x + c * outputChannelOffset;

        var max = float.MinValue;

        for (var ky = 0; ky < layerData.PoolSize; ky++)
            for (var kx = 0; kx < layerData.PoolSize; kx++)
            {
                var oldX = x * layerData.PoolSize + kx;
                var oldY = y * layerData.PoolSize + ky;
                var inputIndex = oldY * layerData.InputWidth + oldX + c * inputChannelOffset;
                max = Math.Max(max, activations[activationInputOffset + inputIndex]);
            }

        activations[activationOutputOffset + outputIndex] = max;

    }

    public static void BackwardKernelImpl(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> activations,
        ArrayView<float> errors)
    {
        int batch = index / (layerData.InputChannels * layerData.OutputHeight * layerData.OutputWidth);
        int c = (index / (layerData.OutputHeight * layerData.OutputWidth)) % layerData.InputChannels;
        int y = (index / layerData.OutputWidth) % layerData.OutputHeight;
        int x = index % layerData.OutputWidth;

        var activationInputOffset = batch * networkData.ActivationCount + layerData.ActivationInputOffset;
        var currentErrorOffset = batch * networkData.ErrorCount + layerData.CurrentLayerErrorOffset;
        var nextErrorOffset = batch * networkData.ErrorCount + layerData.NextLayerErrorOffset;

        var inputChannelOffset = layerData.InputWidth * layerData.InputHeight;
        var outputChannelOffset = layerData.OutputWidth * layerData.OutputHeight;

        var outputIndex = y * layerData.OutputWidth + x + c * outputChannelOffset;

        var max = float.MinValue;
        var maxIndex = -1;

        // Find max position within pool window
        for (var ky = 0; ky < layerData.PoolSize; ky++)
            for (var kx = 0; kx < layerData.PoolSize; kx++)
            {
                var oldX = x * layerData.PoolSize + kx;
                var oldY = y * layerData.PoolSize + ky;
                var inputIndex = oldY * layerData.InputWidth + oldX + c * inputChannelOffset;

                if (activations[activationInputOffset + inputIndex] > max)
                {
                    max = activations[activationInputOffset + inputIndex];
                    maxIndex = inputIndex;
                }
            }

        if (maxIndex >= 0)
            // Propagate error to the max position only
            errors[currentErrorOffset + maxIndex] = errors[nextErrorOffset + outputIndex];
    }
    public static void GradientAccumulationKernelImpl(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> parameters,
        ArrayView<float> gradients)
    {

    }

    public void AccumulateGradients(NetworkAccelerator accelerator)
    {

    }

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>> ForwardKernel { get; private set; }

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>,
        ArrayView<float>> BackwardKernel
    { get; private set; }
    
    public void CompileKernels(Accelerator accelerator)
    {
        ForwardKernel =
            accelerator.LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>>(
                ForwardKernelImpl);
        BackwardKernel =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>>(BackwardKernelImpl);

    }
    public LayerData LayerData { get; set; }

}