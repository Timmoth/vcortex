using ILGPU.Runtime;
using ILGPU;
using vcortex.Accelerated;

namespace vcortex.Layers.Convolution;

public class ReLUConvolutionLayer : IConvolutionalLayer
{
    public ReLUConvolutionLayer()
    {

    }

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
        };
    }

    public void Forward(float[] activations)
    {

    }

    public int GradientCount => 0;

    public void Backward(float[] activations, float[] errors,
        float[] gradients, float learningRate)
    {
     
    }

    public void AccumulateGradients(float[][] gradients, float learningRate)
    {
    }

    public void Forward(NetworkAccelerator accelerator)
    {
        ForwardKernel(accelerator.Network.NetworkData.BatchSize * LayerData.NumOutputs, accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Parameters.View, accelerator.Buffers.Activations.View);
    }

    public void Backward(NetworkAccelerator accelerator)
    {
        for (int i = 0; i < accelerator.Network.NetworkData.BatchSize; i++)
        {
            accelerator.Buffers.Errors.View.SubView(accelerator.Network.NetworkData.ErrorCount * i + LayerData.CurrentLayerErrorOffset, LayerData.NumInputs).MemSetToZero();
        }

        BackwardKernel(accelerator.Network.NetworkData.BatchSize * LayerData.NumOutputs, accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Parameters.View, accelerator.Buffers.Activations.View, accelerator.Buffers.Gradients.View,
            accelerator.Buffers.Errors.View);
    }

    public static void ForwardKernelImpl(
     Index1D index,
     NetworkData networkData,
     LayerData layerData,
     ArrayView<float> parameters,
     ArrayView<float> activations)
    {
        // index = batches * InputChannels * height * width
        int batch = index / layerData.NumOutputs;
        int outputIndex = index % layerData.NumOutputs;

        var activationInputOffset = batch * networkData.ActivationCount + layerData.ActivationInputOffset;
        var activationOutputOffset = batch * networkData.ActivationCount + layerData.ActivationOutputOffset;

        activations[activationOutputOffset + outputIndex] = Math.Max(0, activations[activationInputOffset + outputIndex]);
    }

    public static void BackwardKernelImpl(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> parameters,
        ArrayView<float> activations,
        ArrayView<float> gradients,
        ArrayView<float> errors)
    {
        int batch = index / layerData.NumOutputs;
        int outputIndex = index % layerData.NumOutputs;

        var activationInputOffset = batch * networkData.ActivationCount + layerData.ActivationInputOffset;
        var currentErrorOffset = batch * networkData.ErrorCount + layerData.CurrentLayerErrorOffset;
        var nextErrorOffset = batch * networkData.ErrorCount + layerData.NextLayerErrorOffset;

        errors[currentErrorOffset + outputIndex] = activations[activationInputOffset + outputIndex] > 0 ? errors[nextErrorOffset + outputIndex] : 0;
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

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>> ForwardKernel { get; private set; }

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> BackwardKernel
    { get; private set; }

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>> GradientAccumulationKernel { get; private set; }


    public void CompileKernels(Accelerator accelerator)
    {
        ForwardKernel =
            accelerator.LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>>(
                ForwardKernelImpl);
        BackwardKernel =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(BackwardKernelImpl);
        GradientAccumulationKernel =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>>(GradientAccumulationKernelImpl);
    }
    public LayerData LayerData { get; private set; }

}