using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using vcortex.Layers;
using vcortex.Network;

namespace vcortex.gpu.Layers;

public class DenseLayer : IConnectedLayer
{
    private readonly Dense _dense;
    private readonly BackwardKernelInputs _backwardKernelInputs;
    private readonly ForwardKernelInputs _forwardKernelInputs;
    private readonly NetworkBuffers _buffers;
    private readonly Accelerator _accelerator;

    public DenseLayer(Dense dense, NetworkBuffers buffers, Accelerator accelerator, NetworkData networkData)
    {
        _dense = dense;
        _buffers = buffers;
        _accelerator = accelerator;

        _forwardKernelInputs = new ForwardKernelInputs
        {
            NumInputs = _dense.NumInputs,
            ParameterOffset = _dense.ParameterOffset,
            ActivationCount = networkData.ActivationCount,
            ActivationInputOffset = _dense.ActivationInputOffset,
            NumOutputs = _dense.NumOutputs,
            BiasOffset = _dense.BiasOffset,
            ActivationOutputOffset = _dense.ActivationOutputOffset,
            ActivationType = (int)_dense.Activation
        };
        _backwardKernelInputs = new BackwardKernelInputs
        {
            NumInputs = _dense.NumInputs,
            ParameterOffset = _dense.ParameterOffset,
            ActivationCount = networkData.ActivationCount,
            ActivationInputOffset = _dense.ActivationInputOffset,
            NumOutputs = _dense.NumOutputs,
            BiasOffset = _dense.BiasOffset,
            ActivationOutputOffset = _dense.ActivationOutputOffset,
            CurrentLayerErrorOffset = _dense.CurrentLayerErrorOffset,
            ParameterCount = networkData.ParameterCount,
            NextLayerErrorOffset = _dense.NextLayerErrorOffset,
            ActivationType = (int)_dense.Activation
        };

        _forwardKernel =
            _accelerator
                .LoadAutoGroupedStreamKernel<Index1D, ForwardKernelInputs, ArrayView<float>, ArrayView<float>>(
                    ForwardKernelImpl);
        _backwardKernel1 =
            _accelerator
                .LoadAutoGroupedStreamKernel<Index1D, BackwardKernelInputs, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(BackwardKernel1Impl);

        _backwardKernel2 =
            _accelerator
                .LoadAutoGroupedStreamKernel<Index1D, BackwardKernelInputs, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(BackwardKernel2Impl);

    }

    public Layer Config => _dense;

    public struct ForwardKernelInputs
    {
        public int NumInputs { get; set; }
        public int ParameterOffset { get; set; }
        public int ActivationCount { get; set; }
        public int ActivationInputOffset { get; set; }
        public int NumOutputs { get; set; }
        public int BiasOffset { get; set; }
        public int ActivationOutputOffset { get; set; }
        public int ActivationType { get; set; }
    }

    public struct BackwardKernelInputs
    {
        public int NumInputs { get; set; }
        public int ParameterOffset { get; set; }
        public int ActivationCount { get; set; }
        public int ActivationInputOffset { get; set; }
        public int NumOutputs { get; set; }
        public int BiasOffset { get; set; }
        public int ActivationOutputOffset { get; set; }
        public int CurrentLayerErrorOffset { get; set; }
        public int ParameterCount { get; set; }
        public int NextLayerErrorOffset { get; set; }
        public int ActivationType { get; set; }
    }
    
    #region Kernel

    private readonly Action<Index1D, ForwardKernelInputs, ArrayView<float>, ArrayView<float>> _forwardKernel;

    private readonly Action<Index1D, BackwardKernelInputs, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> _backwardKernel1;

    private readonly Action<Index1D, BackwardKernelInputs, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> _backwardKernel2;

    public virtual void FillRandom()
    {
        var parameters = new float[_dense.ParameterCount];

        var rnd = Random.Shared;
        var variance = 1.0f / _dense.ParameterCount;
        for (var i = 0; i < _dense.ParameterCount; i++)
            parameters[i] = (float)(rnd.NextDouble() * 2 - 1) * MathF.Sqrt(variance);

        _buffers.Parameters.View.SubView(_dense.ParameterOffset, _dense.ParameterCount).CopyFromCPU(parameters);
    }

    public void Forward()
    {
        _forwardKernel(_dense.NumOutputs * _buffers.BatchSize, _forwardKernelInputs, _buffers.Parameters.View,
            _buffers.Activations.View);
    }

    public void Backward()
    {
        _backwardKernel1(_dense.NumOutputs * _dense.NumInputs * _buffers.BatchSize,
            _backwardKernelInputs, _buffers.Parameters.View,
            _buffers.Activations.View, _buffers.Gradients.View,
            _buffers.Errors.View);
        _accelerator.Synchronize();
        _backwardKernel2(_dense.NumOutputs * _buffers.BatchSize,
            _backwardKernelInputs, _buffers.Activations.View,
            _buffers.Gradients.View,
            _buffers.Errors.View);
    }

    public bool IsTraining { get; set; }

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
            // Sigmoid activation function: 1 / (1 + e^(-x))
            activation = 1.0f / (1.0f + XMath.Exp(-sum));
        else if (inputs.ActivationType == 1)
            // ReLU activation function: max(0, x)
            activation = XMath.Max(0.0f, sum);
        else if (inputs.ActivationType == 2)
            // Leaky ReLU activation function: max(alpha * x, x)
            activation = XMath.Max(0.1f * sum, sum);

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
            // Sigmoid derivative: x * (1 - x) where x is the sigmoid output
            derivative = x * (1.0f - x);
        else if (inputs.ActivationType == 1)
            // ReLU derivative: 1 if x > 0, else 0
            derivative = x > 0 ? 1.0f : 0.0f;
        else if (inputs.ActivationType == 2)
            // Leaky ReLU derivative: 1 if x > 0, else alpha
            derivative = x > 0 ? 1.0f : 0.01f;

        var delta = errors[nextErrorOffset + outputIndex] * derivative;

        // Compute the offset for weights of this output neuron
        var weightIndex = outputIndex * inputs.NumInputs + inputIndex;

        // Atomic update to propagate error back to current layer and accumulate weight gradient
        Atomic.Add(ref errors[currentErrorOffset + inputIndex],
            delta * parameters[inputs.ParameterOffset + weightIndex]);

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
            // Sigmoid derivative: x * (1 - x) where x is the sigmoid output
            derivative = x * (1.0f - x);
        else if (inputs.ActivationType == 1)
            // ReLU derivative: 1 if x > 0, else 0
            derivative = x > 0 ? 1.0f : 0.0f;
        else if (inputs.ActivationType == 2)
            // Leaky ReLU derivative: 1 if x > 0, else alpha
            derivative = x > 0 ? 1.0f : 0.01f;

        var delta = errors[nextErrorOffset + outputIndex] * derivative;

        // Update bias gradient for this neuron
        gradients[gradientOffset + inputs.BiasOffset + outputIndex] = delta;
    }

    #endregion
}