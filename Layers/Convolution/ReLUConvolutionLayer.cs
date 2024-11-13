using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using vcortex.Accelerated;

namespace vcortex.Layers.Convolution;

public class ReLUConvolutionLayer : IConvolutionalLayer
{
    public int OutputWidth => InputWidth;
    public int OutputHeight => InputHeight;

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

    public int GradientCount => 0;

    public LayerData LayerData { get; set; }

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
            OutputHeight, InputChannels, OutputChannels, 0, 0, 0);
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
            OutputHeight, InputChannels, OutputChannels, 0, 0, 0);
    }

    #region Kernels

    private Action<Index1D, NetworkData, LayerData, ArrayView<float>> _forwardKernel;
    private Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>> _backwardKernel;
    
    public void Forward(NetworkAccelerator accelerator)
    {
        _forwardKernel(accelerator.Network.NetworkData.BatchSize * LayerData.NumOutputs, accelerator.Network.NetworkData,
            LayerData, accelerator.Buffers.Activations.View);
    }

    public void Backward(NetworkAccelerator accelerator)
    {
        _backwardKernel(accelerator.Network.NetworkData.BatchSize * LayerData.NumOutputs,
            accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Activations.View,
            accelerator.Buffers.Errors.View);
    }

    public void AccumulateGradients(NetworkAccelerator accelerator)
    {
        // No gradients
    }


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
    
    public static void ForwardKernel(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> activations)
    {
        // index = batches * outputs
        int batch = index / layerData.NumOutputs;
        var outputIndex = index % layerData.NumOutputs;

        var activationInputOffset = batch * networkData.ActivationCount + layerData.ActivationInputOffset;
        var activationOutputOffset = batch * networkData.ActivationCount + layerData.ActivationOutputOffset;

        activations[activationOutputOffset + outputIndex] =
            XMath.Max(0, activations[activationInputOffset + outputIndex]);
    }

    public static void BackwardKernel(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> activations,
        ArrayView<float> errors)
    {
        int batch = index / layerData.NumOutputs;
        var outputIndex = index % layerData.NumOutputs;

        var activationInputOffset = batch * networkData.ActivationCount + layerData.ActivationInputOffset;
        var currentErrorOffset = batch * networkData.ErrorCount + layerData.CurrentLayerErrorOffset;
        var nextErrorOffset = batch * networkData.ErrorCount + layerData.NextLayerErrorOffset;

        var activationValue = activations[activationInputOffset + outputIndex];
        errors[currentErrorOffset + outputIndex] = activationValue > 1e-6f ? errors[nextErrorOffset + outputIndex] : 0;
    }

    #endregion
}