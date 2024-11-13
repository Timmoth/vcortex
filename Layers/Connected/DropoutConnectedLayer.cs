using ILGPU;
using ILGPU.Algorithms.Random;
using ILGPU.Runtime;
using vcortex.Accelerated;
using vcortex.Layers.Connected;

namespace vcortex.Layers.Convolution;

public class DropoutConnectedLayer : IConnectedLayer
{

    public DropoutConnectedLayer(float dropoutRate)
    {
        DropoutRate = dropoutRate;
    }

    public float DropoutRate { get; set; }
    public int NumInputs { get; private set; }
    public int NumOutputs { get; private set; }
    public int ActivationInputOffset { get; private set; }
    public int ActivationOutputOffset { get; private set; }
    public int CurrentLayerErrorOffset { get; private set; }
    public int NextLayerErrorOffset { get; private set; }
    public int GradientOffset { get; private set; }
    public int ParameterCount { get; private set; }
    public int ParameterOffset { get; private set; }
    public int GradientCount => 0;

    public LayerData LayerData { get; set; }

    public void Connect(ILayer prevLayer)
    {
        NumInputs = NumOutputs = prevLayer.NumOutputs;

        ParameterCount = 0;
        ParameterOffset = prevLayer.ParameterOffset + prevLayer.ParameterCount;

        ActivationInputOffset = prevLayer.ActivationOutputOffset;
        ActivationOutputOffset = prevLayer.ActivationOutputOffset + prevLayer.NumOutputs;
        CurrentLayerErrorOffset = prevLayer.NextLayerErrorOffset;
        NextLayerErrorOffset = prevLayer.CurrentLayerErrorOffset + NumInputs;
        GradientOffset = prevLayer.GradientOffset + prevLayer.GradientCount;

        LayerData = new LayerData(NumInputs, NumOutputs, ActivationInputOffset, ActivationOutputOffset, GradientOffset,
            NextLayerErrorOffset, CurrentLayerErrorOffset, ParameterOffset, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    }

    public void Connect(ConnectedInputConfig config)
    {
        NumInputs = NumOutputs = config.NumInputs;

        ParameterCount = 0;
        ParameterOffset = 0;

        ActivationInputOffset = 0;
        ActivationOutputOffset = config.NumInputs;
        CurrentLayerErrorOffset = 0;
        NextLayerErrorOffset = config.NumInputs;
        GradientOffset = 0;

        LayerData = new LayerData(NumInputs, NumOutputs, ActivationInputOffset, ActivationOutputOffset, GradientOffset,
            NextLayerErrorOffset, CurrentLayerErrorOffset, ParameterOffset, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    }

    #region Kernels

    private Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, float> _forwardKernel;
    private Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, float> _backwardKernel;
    public MemoryBuffer1D<float, Stride1D.Dense> Mask;
    public RNG<XorShift64Star> Rng;
    public void Forward(NetworkAccelerator accelerator)
    {
        Rng.FillUniform(accelerator.accelerator.DefaultStream, Mask.View);

        _forwardKernel(accelerator.Network.NetworkData.BatchSize * LayerData.NumOutputs, accelerator.Network.NetworkData,
            LayerData, accelerator.Buffers.Activations.View, Mask.View, accelerator.IsTraining ? DropoutRate : 0.0f);
    }

    public void Backward(NetworkAccelerator accelerator)
    {
        _backwardKernel(accelerator.Network.NetworkData.BatchSize * LayerData.NumOutputs,
            accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Errors.View, Mask.View, accelerator.IsTraining ? DropoutRate : 0.0f);
    }

    public void AccumulateGradients(NetworkAccelerator accelerator)
    {
        // No gradients
    }


    public void CompileKernels(NetworkAccelerator accelerator)
    {
        Mask = accelerator.accelerator.Allocate1D<float>(LayerData.NumOutputs);

        var random = new Random();
        Rng = RNG.Create<XorShift64Star>(accelerator.accelerator, random);

        _forwardKernel =
            accelerator.accelerator.LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, float>(
                ForwardKernel);
        _backwardKernel =
            accelerator.accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, float>(
                    BackwardKernel);
    }

    public static void ForwardKernel(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> activations,
        ArrayView<float> mask,
        float droprate)
    {

        // index = batches * outputs
        int batch = index / layerData.NumOutputs;
        var outputIndex = index % layerData.NumOutputs;

        var activationInputOffset = batch * networkData.ActivationCount + layerData.ActivationInputOffset;
        var activationOutputOffset = batch * networkData.ActivationCount + layerData.ActivationOutputOffset;

        // Apply mask: if neuron is kept, use the input; otherwise, set output to zero
        activations[activationOutputOffset + outputIndex] = mask[outputIndex] >= droprate ? activations[activationInputOffset + outputIndex] : 0.0f;
    }

    public static void BackwardKernel(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> errors,
        ArrayView<float> mask,
        float droprate
        )
    {
        int batch = index / layerData.NumOutputs;
        var outputIndex = index % layerData.NumOutputs;

        var currentErrorOffset = batch * networkData.ErrorCount + layerData.CurrentLayerErrorOffset;
        var nextErrorOffset = batch * networkData.ErrorCount + layerData.NextLayerErrorOffset;

        // Only propagate errors for active neurons (those that were not dropped)
        errors[currentErrorOffset + outputIndex] = mask[outputIndex] >= droprate ? errors[nextErrorOffset + outputIndex] : 0.0f;
    }

    #endregion
}