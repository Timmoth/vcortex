using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using vcortex.Core.Layers;

namespace vcortex.gpu.Layers;

public class KernelConvolutionLayer : IConvolutionalLayer
{
    private readonly Convolution _convolution;
    private BackwardKernelInputs _backwardKernelInputs;

    private ForwardKernelInputs _forwardKernelInputs;

    public KernelConvolutionLayer(Convolution convolution)
    {
        _convolution = convolution;
    }

    public Layer Config => _convolution;

    public virtual void FillRandom(INetworkAgent agent)
    {
        var parameters = new float[_convolution.ParameterCount];

        var rnd = Random.Shared;
        var variance = 2.0f / _convolution.ParameterCount;
        for (var i = 0; i < _convolution.ParameterCount; i++)
            parameters[i] = (float)(rnd.NextDouble() * 2 - 1) * MathF.Sqrt(variance);

        agent.Buffers.Parameters.View.SubView(_convolution.ParameterOffset, _convolution.ParameterCount)
            .CopyFromCPU(parameters);
    }

    public void Forward(INetworkAgent agent)
    {
        _forwardKernel(
            agent.Buffers.BatchSize * _convolution.KernelsPerChannel * _convolution.OutputHeight *
            _convolution.OutputWidth * _convolution.InputChannels, _forwardKernelInputs,
            agent.Buffers.Parameters.View, agent.Buffers.Activations.View);
    }

    public void Backward(NetworkTrainer trainer)
    {
        _backwardKernel(
            trainer.Buffers.BatchSize * _convolution.KernelsPerChannel * _convolution.OutputHeight *
            _convolution.OutputWidth * _convolution.InputChannels, _backwardKernelInputs,
            trainer.Buffers.Parameters.View, trainer.Buffers.Activations.View,
            trainer.Buffers.Gradients.View,
            trainer.Buffers.Errors.View);
    }

    public struct ForwardKernelInputs
    {
        public int ActivationInputOffset { get; set; }
        public int ActivationOutputOffset { get; set; }
        public int ParameterOffset { get; set; }
        public int InputWidth { get; set; }
        public int InputHeight { get; set; }
        public int OutputWidth { get; set; }
        public int OutputHeight { get; set; }
        public int InputChannels { get; set; }
        public int NumKernels { get; set; }
        public int KernelSize { get; set; }
        public int Stride { get; set; }
        public int Padding { get; set; }
        public int ActivationCount { get; set; }
        public int ActivationType { get; set; }
    }

    public struct BackwardKernelInputs
    {
        public int ActivationInputOffset { get; set; }
        public int NextLayerErrorOffset { get; set; }
        public int CurrentLayerErrorOffset { get; set; }
        public int ParameterOffset { get; set; }
        public int InputWidth { get; set; }
        public int InputHeight { get; set; }
        public int OutputWidth { get; set; }
        public int OutputHeight { get; set; }
        public int InputChannels { get; set; }
        public int NumKernels { get; set; }
        public int KernelSize { get; set; }
        public int Stride { get; set; }
        public int Padding { get; set; }
        public int ActivationCount { get; set; }
        public int ParameterCount { get; set; }
        public int ActivationType { get; set; }
    }

    #region Kernels

    private Action<Index1D, ForwardKernelInputs, ArrayView<float>, ArrayView<float>> _forwardKernel;

    private Action<Index1D, BackwardKernelInputs, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> _backwardKernel;

    public void CompileKernels(INetworkAgent agent)
    {
        _forwardKernelInputs = new ForwardKernelInputs
        {
            ParameterOffset = _convolution.ParameterOffset,
            ActivationCount = agent.Network.NetworkData.ActivationCount,
            ActivationInputOffset = _convolution.ActivationInputOffset,
            ActivationOutputOffset = _convolution.ActivationOutputOffset,
            ActivationType = (int)_convolution.Activation,
            InputWidth = _convolution.InputWidth,
            InputHeight = _convolution.InputHeight,
            OutputWidth = _convolution.OutputWidth,
            OutputHeight = _convolution.OutputHeight,
            InputChannels = _convolution.InputChannels,
            NumKernels = _convolution.KernelsPerChannel,
            KernelSize = _convolution.KernelSize,
            Stride = _convolution.Stride,
            Padding = _convolution.Padding
        };
        _backwardKernelInputs = new BackwardKernelInputs
        {
            ParameterOffset = _convolution.ParameterOffset,
            ActivationCount = agent.Network.NetworkData.ActivationCount,
            ActivationInputOffset = _convolution.ActivationInputOffset,
            CurrentLayerErrorOffset = _convolution.CurrentLayerErrorOffset,
            ParameterCount = agent.Network.NetworkData.ParameterCount,
            NextLayerErrorOffset = _convolution.NextLayerErrorOffset,
            ActivationType = (int)_convolution.Activation,
            InputWidth = _convolution.InputWidth,
            InputHeight = _convolution.InputHeight,
            OutputWidth = _convolution.OutputWidth,
            OutputHeight = _convolution.OutputHeight,
            InputChannels = _convolution.InputChannels,
            NumKernels = _convolution.KernelsPerChannel,
            KernelSize = _convolution.KernelSize,
            Stride = _convolution.Stride,
            Padding = _convolution.Padding
        };

        _forwardKernel =
            agent.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, ForwardKernelInputs, ArrayView<float>, ArrayView<float>>(
                    ForwardKernelImpl);
        _backwardKernel =
            agent.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, BackwardKernelInputs, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(BackwardKernelImpl);
    }


    public static void ForwardKernelImpl(
        Index1D index,
        ForwardKernelInputs inputs,
        ArrayView<float> parameters,
        ArrayView<float> activations)
    {
        var batch = index / (inputs.NumKernels * inputs.OutputHeight * inputs.OutputWidth *
                             inputs.InputChannels);
        var k = index / (inputs.OutputHeight * inputs.OutputWidth * inputs.InputChannels) %
                inputs.NumKernels;
        var y = index / (inputs.OutputWidth * inputs.InputChannels) % inputs.OutputHeight;
        var x = index / inputs.InputChannels % inputs.OutputWidth;
        var ic = index % inputs.InputChannels;
        var oc = ic * inputs.NumKernels + k;
        var ic_pixel_offset = ic * inputs.InputWidth * inputs.InputHeight;

        // Offsets in the input/output activations
        var activationInputOffset = batch * inputs.ActivationCount + inputs.ActivationInputOffset;
        var activationOutputOffset = batch * inputs.ActivationCount + inputs.ActivationOutputOffset;
        var kernelSize = inputs.KernelSize * inputs.KernelSize;

        // Kernel offset to access weights for this specific kernel and input channel
        var kernelOffset = inputs.ParameterOffset +
                           oc * kernelSize;

        // Accumulate the weighted sum of inputs within the convolutional kernel window
        float sum = 0;
        for (var j = 0; j < kernelSize; j++)
        {
            // Convert linear kernel index to 2D (kernelY, kernelX) coordinates
            var kernelY = j / inputs.KernelSize;
            var kernelX = j % inputs.KernelSize;

            // Adjust input coordinates to account for stride and padding
            var inputY = y * inputs.Stride + kernelY - inputs.Padding;
            var inputX = x * inputs.Stride + kernelX - inputs.Padding;

            // Check for out-of-bounds input coordinates due to padding
            if (inputY >= 0 && inputY < inputs.InputHeight && inputX >= 0 && inputX < inputs.InputWidth)
            {
                var pixelIndex = ic_pixel_offset + inputY * inputs.InputWidth + inputX;
                sum += activations[activationInputOffset + pixelIndex] * parameters[kernelOffset + j];
            }
        }

        // Calculate total output position for this batch, kernel, height, width, and channel
        var outputIndex = y * inputs.OutputWidth + x + oc * inputs.OutputWidth * inputs.OutputHeight;

        // Store the result in the output activations
        if (inputs.ActivationType == 0)
            // Sigmoid activation function: 1 / (1 + e^(-x))
            activations[activationOutputOffset + outputIndex] = 1.0f / (1.0f + XMath.Exp(-sum));
        else if (inputs.ActivationType == 1)
            // ReLU activation function: max(0, x)
            activations[activationOutputOffset + outputIndex] = XMath.Max(0.0f, sum);
        else if (inputs.ActivationType == 2)
            // Leaky ReLU activation function: max(alpha * x, x)
            activations[activationOutputOffset + outputIndex] = XMath.Max(0.1f * sum, sum);
    }

    public static void BackwardKernelImpl(
        Index1D index,
        BackwardKernelInputs inputs,
        ArrayView<float> parameters,
        ArrayView<float> activations,
        ArrayView<float> gradients,
        ArrayView<float> errors)
    {
        // Calculate batch, kernel, output height (y), width (x), and input channel
        int batch = index / (inputs.NumKernels * inputs.OutputHeight * inputs.OutputWidth *
                             inputs.InputChannels);
        var k = index / (inputs.OutputHeight * inputs.OutputWidth * inputs.InputChannels) %
                inputs.NumKernels;
        var y = index / (inputs.OutputWidth * inputs.InputChannels) % inputs.OutputHeight;
        var x = index / inputs.InputChannels % inputs.OutputWidth;
        var ic = index % inputs.InputChannels;
        var oc = ic * inputs.NumKernels + k;
        var ic_pixel_offset = ic * inputs.InputWidth * inputs.InputHeight;

        // Offsets for activations, errors, and gradients
        var activationInputOffset = batch * inputs.ActivationCount + inputs.ActivationInputOffset;
        var currentErrorOffset = batch * inputs.ActivationCount + inputs.CurrentLayerErrorOffset;
        var nextErrorOffset = batch * inputs.ActivationCount + inputs.NextLayerErrorOffset;
        var gradientOffset = batch * inputs.ParameterCount + inputs.ParameterOffset;

        // Compute the output index for error propagation
        var outputIndex = y * inputs.OutputWidth + x + oc * inputs.OutputWidth * inputs.OutputHeight;

        // Error from the next layer for this output position
        var error = errors[nextErrorOffset + outputIndex];

        var kernelSize = inputs.KernelSize * inputs.KernelSize;
        // Start position for this kernel's weights in the parameter array
        var kernelOffset = inputs.ParameterOffset +
                           oc * kernelSize;

        // Accumulate gradients and propagate error to the input layer
        for (var j = 0; j < kernelSize; j++)
        {
            var kernelY = j / inputs.KernelSize;
            var kernelX = j % inputs.KernelSize;

            var inputY = y * inputs.Stride + kernelY - inputs.Padding;
            var inputX = x * inputs.Stride + kernelX - inputs.Padding;

            if (inputY >= 0 && inputY < inputs.InputHeight && inputX >= 0 && inputX < inputs.InputWidth)
            {
                var pixelIndex = ic_pixel_offset + inputY * inputs.InputWidth + inputX;

                // Perform the backward pass logic with the adjusted pixelIndex
                var outputActivation = activations[activationInputOffset + pixelIndex];
                var derivative = 0.0f;
                if (inputs.ActivationType == 0)
                    // Sigmoid: Derivative of sigmoid is sigmoid(x) * (1 - sigmoid(x))
                    derivative = outputActivation * (1.0f - outputActivation);
                else if (inputs.ActivationType == 1)
                    // ReLU: Derivative of ReLU is 1 if x > 0, otherwise 0
                    derivative = outputActivation > 0 ? 1.0f : 0.0f;
                else if (inputs.ActivationType == 2)
                    // Leaky ReLU: Derivative of Leaky ReLU is alpha if x < 0, otherwise 1
                    derivative = outputActivation > 0 ? 1.0f : 0.1f;

                var delta = error * derivative;

                // Accumulate the gradient
                gradients[gradientOffset + oc * kernelSize + j] =
                    delta * activations[activationInputOffset + pixelIndex];

                // Propagate error to the current layer
                Atomic.Add(ref errors[currentErrorOffset + pixelIndex], delta * parameters[kernelOffset + j]);
            }
        }
    }

    #endregion
}