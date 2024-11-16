using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using vcortex.Layers;
using vcortex.Network;

namespace vcortex.gpu.Layers;

public class SoftmaxConnectedLayer : IConnectedLayer
{
    private readonly Softmax _softmax;
    private readonly BackwardKernelInputs _backwardKernelInputs;

    private readonly ForwardKernelInputs _forwardKernelInputs;

    private readonly NetworkBuffers _buffers;
    private readonly Accelerator _accelerator;
    public SoftmaxConnectedLayer(Softmax softmax, NetworkBuffers buffers, Accelerator accelerator, NetworkData networkData)
    {
        _buffers = buffers;
        _accelerator = accelerator;
        _softmax = softmax;

        _forwardKernelInputs = new ForwardKernelInputs
        {
            ActivationCount = networkData.ActivationCount,
            ActivationInputOffset = _softmax.ActivationInputOffset,
            NumOutputs = _softmax.NumOutputs,
            ActivationOutputOffset = _softmax.ActivationOutputOffset,
            NumInputs = _softmax.NumInputs,
            ParameterOffset = _softmax.ParameterOffset,
            BiasOffset = _softmax.BiasOffset
        };
        _backwardKernelInputs = new BackwardKernelInputs
        {
            ActivationCount = networkData.ActivationCount,
            NumOutputs = _softmax.NumOutputs,
            CurrentLayerErrorOffset = _softmax.CurrentLayerErrorOffset,
            NextLayerErrorOffset = _softmax.NextLayerErrorOffset,
            BiasOffset = _softmax.BiasOffset,
            NumInputs = _softmax.NumInputs,
            ActivationInputOffset = _softmax.ActivationInputOffset,
            ParameterCount = networkData.ParameterCount,
            ParameterOffset = _softmax.ParameterOffset
        };

        _forwardKernel1 =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, ForwardKernelInputs, ArrayView<float>, ArrayView<float>>(
                    ForwardKernel1Impl);

        _forwardKernel2 =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, ForwardKernelInputs, ArrayView<float>, ArrayView<float>>(
                    ForwardKernel2Impl);

        _backwardKernel1 =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, BackwardKernelInputs, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(BackwardKernel1Impl);

        _backwardKernel2 =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, BackwardKernelInputs, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(BackwardKernel2Impl);
    }

    public Layer Config => _softmax;

    public virtual void FillRandom()
    {
        var parameters = new float[_softmax.ParameterCount];

        var rnd = Random.Shared;
        var variance = 1.0f / _softmax.ParameterCount;
        for (var i = 0; i < _softmax.ParameterCount; i++)
            parameters[i] = (float)(rnd.NextDouble() * 2 - 1) * MathF.Sqrt(variance);

        _buffers.Parameters.View.SubView(_softmax.ParameterOffset, _softmax.ParameterCount)
            .CopyFromCPU(parameters);
    }

    public void Forward()
    {
        _forwardKernel1(_softmax.NumOutputs * _buffers.BatchSize,
            _forwardKernelInputs, _buffers.Parameters.View,
            _buffers.Activations.View);
        _accelerator.Synchronize();

        _forwardKernel2(_buffers.BatchSize, _forwardKernelInputs,
            _buffers.Parameters.View, _buffers.Activations.View);
    }

    public void Backward()
    {
        _backwardKernel1(_softmax.NumOutputs * _softmax.NumInputs * _buffers.BatchSize,
            _backwardKernelInputs, _buffers.Parameters.View,
            _buffers.Activations.View, _buffers.Gradients.View,
            _buffers.Errors.View);
        _accelerator.Synchronize();

        _backwardKernel2(_softmax.NumOutputs * _buffers.BatchSize,
            _backwardKernelInputs, _buffers.Parameters.View,
            _buffers.Activations.View, _buffers.Gradients.View,
            _buffers.Errors.View);
    }

    public bool IsTraining { get; set; }

    public struct ForwardKernelInputs
    {
        public int ActivationCount { get; set; }
        public int ActivationInputOffset { get; set; }
        public int NumOutputs { get; set; }
        public int ActivationOutputOffset { get; set; }
        public int NumInputs { get; set; }
        public int ParameterOffset { get; set; }
        public int BiasOffset { get; set; }
    }

    public struct BackwardKernelInputs
    {
        public int ActivationCount { get; set; }
        public int NumOutputs { get; set; }
        public int CurrentLayerErrorOffset { get; set; }
        public int NextLayerErrorOffset { get; set; }
        public Index1D BiasOffset { get; set; }
        public int NumInputs { get; set; }
        public int ActivationInputOffset { get; set; }
        public int ParameterCount { get; set; }
        public int ParameterOffset { get; set; }
    }


    #region Kernels

    private readonly Action<Index1D, ForwardKernelInputs, ArrayView<float>, ArrayView<float>> _forwardKernel1;

    private readonly Action<Index1D, ForwardKernelInputs, ArrayView<float>, ArrayView<float>> _forwardKernel2;

    private readonly Action<Index1D, BackwardKernelInputs, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> _backwardKernel1;

    private readonly Action<Index1D, BackwardKernelInputs, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> _backwardKernel2;

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
        Atomic.Add(ref errors[currentErrorOffset + inputIndex],
            delta * parameters[inputs.ParameterOffset + inputs.NumInputs * outputIndex + inputIndex]);

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