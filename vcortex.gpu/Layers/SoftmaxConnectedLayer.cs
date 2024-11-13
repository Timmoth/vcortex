using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using vcortex.Core;
using vcortex.Core.Layers;

namespace vcortex.gpu.Layers;

public class SoftmaxConnectedLayer : IConnectedLayer
{
    public SoftmaxConnectedLayer(int numOutputs)
    {
        NumOutputs = numOutputs;
    }

    public int BiasOffset => NumInputs * NumOutputs;

    public int NumInputs { get; private set; }
    public int NumOutputs { get; }
    public int ActivationInputOffset { get; private set; }
    public int ActivationOutputOffset { get; private set; }
    public int CurrentLayerErrorOffset { get; private set; }
    public int NextLayerErrorOffset { get; private set; }
    public int ParameterCount { get; private set; }
    public int ParameterOffset { get; private set; }
    private ForwardKernelInputs _forwardKernelInputs;
    private BackwardKernelInputs _backwardKernelInputs;
    public void Connect(ILayer prevLayer)
    {
        NumInputs = prevLayer.NumOutputs;

        ParameterCount = NumOutputs * NumInputs + NumOutputs;
        ParameterOffset = prevLayer.ParameterOffset + prevLayer.ParameterCount;

        ActivationInputOffset = prevLayer.ActivationOutputOffset;
        ActivationOutputOffset = prevLayer.ActivationOutputOffset + prevLayer.NumOutputs;
        CurrentLayerErrorOffset = prevLayer.NextLayerErrorOffset;
        NextLayerErrorOffset = prevLayer.CurrentLayerErrorOffset + NumInputs;

        LayerData = new LayerData(NumInputs, NumOutputs, ActivationInputOffset, ActivationOutputOffset,
            NextLayerErrorOffset, CurrentLayerErrorOffset, ParameterOffset, BiasOffset, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    }

    public void Connect(ConnectedInputConfig config)
    {
        NumInputs = config.NumInputs;

        ParameterCount = NumOutputs * NumInputs + NumOutputs;
        ParameterOffset = 0;

        ActivationInputOffset = 0;
        ActivationOutputOffset = config.NumInputs;
        CurrentLayerErrorOffset = 0;
        NextLayerErrorOffset = config.NumInputs;

        LayerData = new LayerData(NumInputs, NumOutputs, ActivationInputOffset, ActivationOutputOffset,
            NextLayerErrorOffset, CurrentLayerErrorOffset, ParameterOffset, BiasOffset, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    }
    
    public struct ForwardKernelInputs
    {
        public required int ActivationCount { get; set; }
        public required int ActivationInputOffset { get; set; }
        public required int NumOutputs { get; set; }
        public required int ActivationOutputOffset { get; set; }
        public required int NumInputs { get; set; }
        public required int ParameterOffset { get; set; }
        public required int BiasOffset { get; set; }
    }
    
    public struct BackwardKernelInputs
    {
        public required int ActivationCount { get; set; }
        public required int NumOutputs { get; set; }
        public required int CurrentLayerErrorOffset { get; set; }
        public required int NextLayerErrorOffset { get; set; }
        public required Index1D BiasOffset { get; set; }
        public required int NumInputs { get; set; }
        public required int ActivationInputOffset { get; set; }
        public required int ParameterCount { get; set; }
        public required int ParameterOffset { get; set; }
    }

    public virtual void FillRandom(INetworkAgent agent)
    {
        var parameters = new float[ParameterCount];

        var rnd = Random.Shared;
        var variance = 1.0f / (ParameterCount);
        for (var i = 0; i < ParameterCount; i++)
        {
            parameters[i] = (float)(rnd.NextDouble() * 2 - 1) * MathF.Sqrt(variance);
        }

        agent.Buffers.Parameters.View.SubView(LayerData.ParameterOffset, ParameterCount).CopyFromCPU(parameters);
    }
    
    public void Forward(INetworkAgent agent)
    {
        _forwardKernel1(LayerData.NumOutputs * agent.Buffers.BatchSize,
            _forwardKernelInputs, agent.Buffers.Parameters.View,
            agent.Buffers.Activations.View);
        agent.Accelerator.Synchronize();

        _forwardKernel2(agent.Buffers.BatchSize, _forwardKernelInputs,
            agent.Buffers.Parameters.View, agent.Buffers.Activations.View);
    }

    public void Backward(NetworkTrainer trainer)
    {
        _backwardKernel1(LayerData.NumOutputs * LayerData.NumInputs * trainer.Buffers.BatchSize,
            _backwardKernelInputs, trainer.Buffers.Parameters.View,
            trainer.Buffers.Activations.View, trainer.Buffers.Gradients.View,
            trainer.Buffers.Errors.View);
        trainer.Accelerator.Synchronize();

        _backwardKernel2(LayerData.NumOutputs * trainer.Buffers.BatchSize,
            _backwardKernelInputs, trainer.Buffers.Parameters.View,
            trainer.Buffers.Activations.View, trainer.Buffers.Gradients.View,
            trainer.Buffers.Errors.View);
    }

    public LayerData LayerData { get; set; }

    #region Kernels


    private Action<Index1D, ForwardKernelInputs, ArrayView<float>, ArrayView<float>> _forwardKernel1;

    private Action<Index1D, ForwardKernelInputs, ArrayView<float>, ArrayView<float>> _forwardKernel2;

    private Action<Index1D, BackwardKernelInputs, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> _backwardKernel1;

    private Action<Index1D, BackwardKernelInputs, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> _backwardKernel2;
    
    
    public void CompileKernels(INetworkAgent agent)
    {
        _forwardKernelInputs = new ForwardKernelInputs
        {
            ActivationCount = agent.Network.NetworkData.ActivationCount,
            ActivationInputOffset = ActivationInputOffset,
            NumOutputs = NumOutputs,
            ActivationOutputOffset = ActivationOutputOffset,
            NumInputs = NumInputs,
            ParameterOffset = ParameterOffset,
            BiasOffset = BiasOffset,
        };
        _backwardKernelInputs = new BackwardKernelInputs
        {
            ActivationCount = agent.Network.NetworkData.ActivationCount,
            NumOutputs = NumOutputs,
            CurrentLayerErrorOffset = CurrentLayerErrorOffset,
            NextLayerErrorOffset = NextLayerErrorOffset,
            BiasOffset = BiasOffset,
            NumInputs = NumInputs,
            ActivationInputOffset = ActivationInputOffset,
            ParameterCount = agent.Network.NetworkData.ParameterCount,
            ParameterOffset = ParameterOffset,
        };    
        
        _forwardKernel1 =
            agent.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, ForwardKernelInputs, ArrayView<float>, ArrayView<float>>(
                    ForwardKernel1Impl);

        _forwardKernel2 =
            agent.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, ForwardKernelInputs, ArrayView<float>, ArrayView<float>>(
                    ForwardKernel2Impl);

        _backwardKernel1 =
            agent.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, BackwardKernelInputs, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(BackwardKernel1Impl);

        _backwardKernel2 =
            agent.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, BackwardKernelInputs, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(BackwardKernel2Impl);
    }

    public static void ForwardKernel1Impl(
        Index1D index,
        ForwardKernelInputs inputs,
        ArrayView<float> parameters,
        ArrayView<float> activations)
    {
        // Calculate output index and batch offset
        var batchIndex = index / inputs.NumOutputs;
        var outputIndex = index % inputs.NumOutputs;

        // Activation offsets
        var activationInputOffset = batchIndex * inputs.ActivationCount + inputs.ActivationInputOffset;
        var activationOutputOffset = batchIndex * inputs.ActivationCount + inputs.ActivationOutputOffset;

        // Initialize the sum with the bias term
        var sum = parameters[inputs.ParameterOffset + inputs.BiasOffset + outputIndex];

        // Offset for weights for this output neuron
        var weightsOffset = inputs.ParameterOffset + inputs.NumInputs * outputIndex;

        // Compute the weighted sum for this output
        for (var j = 0; j < inputs.NumInputs; j++)
            sum += activations[activationInputOffset + j] * parameters[weightsOffset + j];

        // Store the result as the "raw score" in the output activations
        activations[activationOutputOffset + outputIndex] = sum;
    }

    public static void ForwardKernel2Impl(
        Index1D index,
        ForwardKernelInputs inputs,
        ArrayView<float> parameters,
        ArrayView<float> activations)
    {
        var batchOffset = index;
        var activationOutputOffset = batchOffset * inputs.ActivationCount + inputs.ActivationOutputOffset;

        // Find the maximum value in the outputs for numerical stability
        var maxVal = activations[activationOutputOffset];
        for (var i = 1; i < inputs.NumOutputs; i++)
            maxVal = XMath.Max(maxVal, activations[activationOutputOffset + i]);

        // Compute the sum of exponentials
        float sumExp = 0;
        for (var i = 0; i < inputs.NumOutputs; i++)
        {
            activations[activationOutputOffset + i] = XMath.Exp(activations[activationOutputOffset + i] - maxVal);
            sumExp += activations[activationOutputOffset + i];
        }
        
        if (sumExp <= 0 || float.IsNaN(sumExp) || float.IsInfinity(sumExp))
        {
            var uniformProb = 1.0f / inputs.NumOutputs;
            for (var i = 0; i < inputs.NumOutputs; i++) activations[activationOutputOffset + i] = uniformProb;
        }
        else
        {
            // Normalize to get probabilities
            for (var i = 0; i < inputs.NumOutputs; i++)
                activations[activationOutputOffset + i] /= sumExp;
        }
    }

    public static void BackwardKernel1Impl(
        Index1D index,
        BackwardKernelInputs inputs,
        ArrayView<float> parameters,
        ArrayView<float> activations,
        ArrayView<float> gradients,
        ArrayView<float> errors)
    {
        int batchIndex = index / (inputs.NumInputs * inputs.NumOutputs);
        var outputIndex = index / inputs.NumInputs % inputs.NumOutputs;
        var inputIndex = index % inputs.NumInputs;

        // Offsets for activations and errors
        var activationInputOffset = batchIndex * inputs.ActivationCount + inputs.ActivationInputOffset;
        var currentErrorOffset = batchIndex * inputs.ActivationCount + inputs.CurrentLayerErrorOffset;
        var nextErrorOffset = batchIndex * inputs.ActivationCount + inputs.NextLayerErrorOffset;
        var gradientOffset = batchIndex * inputs.ParameterCount + inputs.ParameterOffset; 

        // Calculate delta for this output neuron
        var delta = errors[nextErrorOffset + outputIndex];

        // Compute the gradient contribution for each weight and accumulate errors for backpropagation
        Atomic.Add(ref errors[currentErrorOffset + inputIndex], delta * parameters[inputs.ParameterOffset + inputs.NumInputs * outputIndex + inputIndex]);

        // Store gradient for the current weight
        gradients[gradientOffset + outputIndex * inputs.NumInputs + inputIndex] =
            delta * activations[activationInputOffset + inputIndex];
    }

    public static void BackwardKernel2Impl(
        Index1D index,
        BackwardKernelInputs inputs,
        ArrayView<float> parameters,
        ArrayView<float> activations,
        ArrayView<float> gradients,
        ArrayView<float> errors)
    {
        var batchIndex = index / inputs.NumOutputs;
        var outputIndex = index % inputs.NumOutputs;

        var nextErrorOffset = batchIndex * inputs.ActivationCount + inputs.NextLayerErrorOffset;
        var gradientOffset = batchIndex * inputs.ParameterCount + inputs.ParameterOffset;

        // Calculate gradient for the bias term of this output neuron
        var delta = errors[nextErrorOffset + outputIndex];
        gradients[gradientOffset + inputs.BiasOffset + outputIndex] = delta;
    }

    #endregion
}