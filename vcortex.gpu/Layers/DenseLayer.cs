using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using vcortex.Core;
using vcortex.Core.Layers;

namespace vcortex.gpu.Layers;


public class DenseLayer : IConnectedLayer
{
    public DenseLayer(int numOutputs, ActivationType activationType)
    {
        NumOutputs = numOutputs;
        ActivationType = activationType;
    }

    public ActivationType ActivationType { get; set; }
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
    public struct ForwardKernelInputs
    {
        public required int NumInputs { get; set; }
        public required int ParameterOffset { get; set; }
        public required int ActivationCount { get; set; }
        public required int ActivationInputOffset { get; set; }
        public required int NumOutputs { get; set; }
        public required int BiasOffset { get; set; }
        public required int ActivationOutputOffset { get; set; }
        public required int ActivationType { get; set; }
    }
    
    public struct BackwardKernelInputs
    {
        public required int NumInputs { get; set; }
        public required int ParameterOffset { get; set; }
        public required int ActivationCount { get; set; }
        public required int ActivationInputOffset { get; set; }
        public required int NumOutputs { get; set; }
        public required int BiasOffset { get; set; }
        public required int ActivationOutputOffset { get; set; }
        public required int CurrentLayerErrorOffset { get; set; }
        public required int ParameterCount { get; set; }
        public required int NextLayerErrorOffset { get; set; }
        public required int ActivationType { get; set; }
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

    #region Kernel


    private Action<Index1D, ForwardKernelInputs, ArrayView<float>, ArrayView<float>> _forwardKernel;

    private Action<Index1D, BackwardKernelInputs, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> _backwardKernel1;

    private Action<Index1D, BackwardKernelInputs, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> _backwardKernel2;
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
        _forwardKernel(LayerData.NumOutputs * agent.Buffers.BatchSize,_forwardKernelInputs, agent.Buffers.Parameters.View, agent.Buffers.Activations.View);
    }

    public void Backward(NetworkTrainer trainer)
    {
        _backwardKernel1(LayerData.NumOutputs * LayerData.NumInputs * trainer.Buffers.BatchSize,
            _backwardKernelInputs, trainer.Buffers.Parameters.View,
            trainer.Buffers.Activations.View, trainer.Buffers.Gradients.View,
            trainer.Buffers.Errors.View);
        trainer.Accelerator.Synchronize();
        _backwardKernel2(LayerData.NumOutputs * trainer.Buffers.BatchSize,
            _backwardKernelInputs, trainer.Buffers.Activations.View,
            trainer.Buffers.Gradients.View,
            trainer.Buffers.Errors.View);
    }
    
    public void CompileKernels(INetworkAgent agent)
    {
        _forwardKernelInputs = new ForwardKernelInputs()
        {
            NumInputs = NumInputs,
            ParameterOffset = ParameterOffset,
            ActivationCount = agent.Network.NetworkData.ActivationCount,
            ActivationInputOffset = ActivationInputOffset,
            NumOutputs = NumOutputs,
            BiasOffset = BiasOffset,
            ActivationOutputOffset = ActivationOutputOffset,
            ActivationType = (int)ActivationType
        };
        _backwardKernelInputs = new BackwardKernelInputs()
        {
            NumInputs = NumInputs,
            ParameterOffset = ParameterOffset,
            ActivationCount = agent.Network.NetworkData.ActivationCount,
            ActivationInputOffset = ActivationInputOffset,
            NumOutputs = NumOutputs,
            BiasOffset = BiasOffset,
            ActivationOutputOffset = ActivationOutputOffset,
            CurrentLayerErrorOffset = CurrentLayerErrorOffset,
            ParameterCount = agent.Network.NetworkData.ParameterCount,
            NextLayerErrorOffset = NextLayerErrorOffset,
            ActivationType = (int)ActivationType
        };    
        
        _forwardKernel =
            agent.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, ForwardKernelInputs, ArrayView<float>, ArrayView<float>>(
                    ForwardKernelImpl);
        _backwardKernel1 =
            agent.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, BackwardKernelInputs, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(BackwardKernel1Impl);

        _backwardKernel2 =
            agent.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, BackwardKernelInputs, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(BackwardKernel2Impl);
    }

    public LayerData LayerData { get; set; }

 
    
    // Forward pass kernel for calculating activations in a sigmoid layer
    public static void ForwardKernelImpl(
        Index1D index,
        ForwardKernelInputs inputs,
        ArrayView<float> parameters,
        ArrayView<float> activations)
    {
        // Compute indices for the batch and neuron within the batch
        var batchIndex = index / inputs.NumOutputs;
        var outputIndex = index % inputs.NumOutputs;

        // Calculate offset for input and output activations in the batch
        var activationInputOffset = batchIndex * inputs.ActivationCount + inputs.ActivationInputOffset;
        var activationOutputOffset = batchIndex * inputs.ActivationCount + inputs.ActivationOutputOffset;

        // Start with the bias value as the initial sum for this neuron
        var sum = parameters[inputs.ParameterOffset + inputs.BiasOffset + outputIndex];
        var weightsOffset = inputs.ParameterOffset + outputIndex * inputs.NumInputs;

        // Accumulate the weighted sum for each input
        for (var j = 0; j < inputs.NumInputs; j++)
            sum += activations[activationInputOffset + j] * parameters[weightsOffset + j];

        var activation = 0.0f;
        if (inputs.ActivationType == 0)
        {
            // Sigmoid activation function: 1 / (1 + e^(-x))
            activation = 1.0f / (1.0f + XMath.Exp(-sum));
        }else  if (inputs.ActivationType == 1)
        {
            // ReLU activation function: max(0, x)
            activation = XMath.Max(0.0f, sum);
        }else  if (inputs.ActivationType == 2)
        {
            // Leaky ReLU activation function: max(alpha * x, x)
            activation = XMath.Max(0.1f * sum, sum);
        }
        
        // Apply sigmoid activation and store the result
        activations[activationOutputOffset + outputIndex] = activation;
    }
    
    // Backward pass kernel 1 for calculating weight gradients and propagating errors
    public static void BackwardKernel1Impl(
        Index1D index,
        BackwardKernelInputs inputs,
        ArrayView<float> parameters,
        ArrayView<float> activations,
        ArrayView<float> gradients,
        ArrayView<float> errors)
    {
        // Calculate indices for batch, output, and input
        int batchIndex = index / (inputs.NumInputs * inputs.NumOutputs);
        var outputIndex = index / inputs.NumInputs % inputs.NumOutputs;
        var inputIndex = index % inputs.NumInputs;

        // Calculate relevant offsets
        var activationInputOffset = batchIndex * inputs.ActivationCount + inputs.ActivationInputOffset;
        var activationOutputOffset = batchIndex * inputs.ActivationCount + inputs.ActivationOutputOffset;
        var currentErrorOffset = batchIndex * inputs.ActivationCount + inputs.CurrentLayerErrorOffset;
        var nextErrorOffset = batchIndex * inputs.ActivationCount + inputs.NextLayerErrorOffset;
        var gradientOffset = batchIndex * inputs.ParameterCount + inputs.ParameterOffset;

        // Calculate the derivative of the sigmoid activation
        var x = activations[activationOutputOffset + outputIndex];
        var derivative = 0.0f;
        if (inputs.ActivationType == 0)
        {
            // Sigmoid derivative: x * (1 - x) where x is the sigmoid output
            derivative = x * (1.0f - x);
        }else  if (inputs.ActivationType == 1)
        {
            // ReLU derivative: 1 if x > 0, else 0
            derivative = x > 0 ? 1.0f : 0.0f;
            
        }else  if (inputs.ActivationType == 2)
        {
            // Leaky ReLU derivative: 1 if x > 0, else alpha
            derivative = x > 0 ? 1.0f : 0.01f;
        }
        
        var delta = errors[nextErrorOffset + outputIndex] * derivative;

        // Compute the offset for weights of this output neuron
        var weightIndex = outputIndex * inputs.NumInputs + inputIndex;

        // Atomic update to propagate error back to current layer and accumulate weight gradient
        Atomic.Add(ref errors[currentErrorOffset + inputIndex], delta * parameters[inputs.ParameterOffset + weightIndex]);
        
        gradients[gradientOffset + weightIndex] = delta * activations[activationInputOffset + inputIndex];
    }


    // Backward pass kernel 2 for updating bias gradients
    public static void BackwardKernel2Impl(
        Index1D index,
        BackwardKernelInputs inputs,
        ArrayView<float> activations,
        ArrayView<float> gradients,
        ArrayView<float> errors)
    {
        // Calculate indices and offsets
        var batchIndex = index / inputs.NumOutputs;
        var outputIndex = index % inputs.NumOutputs;
        var activationOutputOffset = batchIndex * inputs.ActivationCount + inputs.ActivationOutputOffset;
        var nextErrorOffset = batchIndex * inputs.ActivationCount + inputs.NextLayerErrorOffset;
        var gradientOffset = batchIndex * inputs.ParameterCount + inputs.ParameterOffset;

        // Compute delta using sigmoid derivative
        var x = activations[activationOutputOffset + outputIndex];
        var derivative = 0.0f;
        if (inputs.ActivationType == 0)
        {
            // Sigmoid derivative: x * (1 - x) where x is the sigmoid output
            derivative = x * (1.0f - x);
        }else  if (inputs.ActivationType == 1)
        {
            // ReLU derivative: 1 if x > 0, else 0
            derivative = x > 0 ? 1.0f : 0.0f;
            
        }else  if (inputs.ActivationType == 2)
        {
            // Leaky ReLU derivative: 1 if x > 0, else alpha
            derivative = x > 0 ? 1.0f : 0.01f;
        }
        
        var delta = errors[nextErrorOffset + outputIndex] * derivative;

        // Update bias gradient for this neuron
        gradients[gradientOffset + inputs.BiasOffset + outputIndex] = delta;
    }

    #endregion
}