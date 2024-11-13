using ILGPU;
using ILGPU.Algorithms.Random;
using ILGPU.Runtime;
using vcortex.Core;
using vcortex.Core.Layers;

namespace vcortex.gpu.Layers;

public class DropoutLayer : IConnectedLayer
{
    public DropoutLayer(float dropoutRate)
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
    public int ParameterCount { get; private set; }
    public int ParameterOffset { get; private set; }

    public LayerData LayerData { get; set; }
    private ForwardKernelInputs _forwardKernelInputs;
    private BackwardKernelInputs _backwardKernelInputs;
    public void Connect(ILayer prevLayer)
    {
        NumInputs = NumOutputs = prevLayer.NumOutputs;

        ParameterCount = 0;
        ParameterOffset = prevLayer.ParameterOffset + prevLayer.ParameterCount;

        ActivationInputOffset = prevLayer.ActivationOutputOffset;
        ActivationOutputOffset = prevLayer.ActivationOutputOffset + prevLayer.NumOutputs;
        CurrentLayerErrorOffset = prevLayer.NextLayerErrorOffset;
        NextLayerErrorOffset = prevLayer.CurrentLayerErrorOffset + NumInputs;

        LayerData = new LayerData(NumInputs, NumOutputs, ActivationInputOffset, ActivationOutputOffset,
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

        LayerData = new LayerData(NumInputs, NumOutputs, ActivationInputOffset, ActivationOutputOffset,
            NextLayerErrorOffset, CurrentLayerErrorOffset, ParameterOffset, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    }
 
    public struct ForwardKernelInputs
    {
        public required int ActivationCount { get; set; }
        public required int ActivationInputOffset { get; set; }
        public required int NumOutputs { get; set; }
        public required int ActivationOutputOffset { get; set; }
        public required float DropRate { get; set; }
    }
    
    public struct BackwardKernelInputs
    {
        public required int ActivationCount { get; set; }
        public required int NumOutputs { get; set; }
        public required int CurrentLayerErrorOffset { get; set; }
        public required int NextLayerErrorOffset { get; set; }
        public required float DropRate { get; set; }
    }
    
    #region Kernels

    private Action<Index1D, ForwardKernelInputs, ArrayView<float>, ArrayView<float>> _forwardKernel;
    private Action<Index1D,BackwardKernelInputs, ArrayView<float>, ArrayView<float>> _backwardKernel;
    private MemoryBuffer1D<float, Stride1D.Dense> Mask;
    private RNG<XorShift64Star> Rng;


    public void FillRandom(INetworkAgent agent)
    {
        
    }

    public void Forward(INetworkAgent agent)
    {
        Rng.FillUniform(agent.Accelerator.DefaultStream, Mask.View);

        _forwardKernelInputs.DropRate = agent.IsTraining ? DropoutRate : 0.0f;
        _forwardKernel(agent.Buffers.BatchSize * LayerData.NumOutputs, _forwardKernelInputs, agent.Buffers.Activations.View, Mask.View);
    }

    public void Backward(NetworkTrainer trainer)
    {      
        _backwardKernelInputs.DropRate = trainer.IsTraining ? DropoutRate : 0.0f;
        _backwardKernel(trainer.Buffers.BatchSize * LayerData.NumOutputs,
            _backwardKernelInputs, trainer.Buffers.Errors.View, Mask.View);
    }

    public void CompileKernels(INetworkAgent agent)
    {
        _forwardKernelInputs = new ForwardKernelInputs()
        {
            ActivationCount = agent.Network.NetworkData.ActivationCount,
            ActivationInputOffset = ActivationInputOffset,
            NumOutputs = NumOutputs,
            ActivationOutputOffset = ActivationOutputOffset,
            DropRate = DropoutRate
        };
        _backwardKernelInputs = new BackwardKernelInputs()
        {
            ActivationCount = agent.Network.NetworkData.ActivationCount,
            NumOutputs = NumOutputs,
            CurrentLayerErrorOffset = CurrentLayerErrorOffset,
            NextLayerErrorOffset = NextLayerErrorOffset,
            DropRate = DropoutRate
        };    
        
        Mask = agent.Accelerator.Allocate1D<float>(LayerData.NumOutputs);

        var random = new Random();
        Rng = RNG.Create<XorShift64Star>(agent.Accelerator, random);

        _forwardKernel =
            agent.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ForwardKernelInputs, ArrayView<float>, ArrayView<float>>(
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
        activations[activationOutputOffset + outputIndex] = mask[outputIndex] >= inputs.DropRate ? activations[activationInputOffset + outputIndex] : 0.0f;
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
        errors[currentErrorOffset + outputIndex] = mask[outputIndex] >= inputs.DropRate ? errors[nextErrorOffset + outputIndex] : 0.0f;
    }

    #endregion
}