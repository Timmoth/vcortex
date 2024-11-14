using vcortex.Layers;
using vcortex.Network;

namespace vcortex.cpu.Layers;

public class KernelConvolutionLayer : IConvolutionalLayer
{
    private readonly Convolution _convolution;
    private readonly NetworkAcceleratorBuffers _buffers;
    private readonly NetworkData _networkData;
    public KernelConvolutionLayer(Convolution convolution, NetworkAcceleratorBuffers buffers, NetworkData networkData)
    {
        _convolution = convolution;
        _buffers = buffers;
        _networkData = networkData;
    }

    public Layer Config => _convolution;
    public void FillRandom()
    {
        var rnd = Random.Shared;
        var variance = 2.0f / _convolution.ParameterCount;
        for (var i = 0; i < _convolution.ParameterCount; i++)
            _buffers.Parameters[_convolution.ParameterOffset +i] = (float)(rnd.NextDouble() * 2 - 1) * MathF.Sqrt(variance);
    }

  public void Forward()
{
    Parallel.For(0, _buffers.BatchSize, batchIndex =>
    {
        // Precompute the input and output activation offsets for this batch
        int activationInputOffset = batchIndex * _networkData.ActivationCount + _convolution.ActivationInputOffset;
        int activationOutputOffset = batchIndex * _networkData.ActivationCount + _convolution.ActivationOutputOffset;
        int kernelSize = _convolution.KernelSize * _convolution.KernelSize;

        // Precompute kernelY and kernelX for each j in the kernel
        Span<int> kernelYs = stackalloc int[kernelSize];
        Span<int> kernelXs = stackalloc int[kernelSize];
        for (var j = 0; j < kernelSize; j++)
        {
            kernelYs[j] = j / _convolution.KernelSize;
            kernelXs[j] = j % _convolution.KernelSize;
        }

        // Loop over kernels per channel and image coordinates
        for (var kernelIndex = 0; kernelIndex < _convolution.KernelsPerChannel; kernelIndex++)
        {
            for (var y = 0; y < _convolution.OutputHeight; y++)
            {
                for (var x = 0; x < _convolution.OutputWidth; x++)
                {
                    for (var ic = 0; ic < _convolution.InputChannels; ic++)
                    {
                        var oc = ic * _convolution.KernelsPerChannel + kernelIndex;
                        var icPixelOffset = ic * _convolution.InputWidth * _convolution.InputHeight;

                        // Kernel offset for weights specific to this kernel and input channel
                        var kernelOffset = _convolution.ParameterOffset + oc * kernelSize;

                        // Accumulate the weighted sum for the kernel window
                        float sum = 0;
                        for (var j = 0; j < kernelSize; j++)
                        {
                            // Use precomputed kernelY and kernelX
                            var kernelY = kernelYs[j];
                            var kernelX = kernelXs[j];

                            // Adjust input coordinates for stride and padding
                            var inputY = y * _convolution.Stride + kernelY - _convolution.Padding;
                            var inputX = x * _convolution.Stride + kernelX - _convolution.Padding;

                            // Only add valid (inputY, inputX) values to avoid bounds checks in the innermost loop
                            if (inputY < 0 || inputY >= _convolution.InputHeight ||
                                inputX < 0 || inputX >= _convolution.InputWidth) continue;
                            var pixelIndex = icPixelOffset + inputY * _convolution.InputWidth + inputX;
                            sum += _buffers.Activations[activationInputOffset + pixelIndex] * 
                                   _buffers.Parameters[kernelOffset + j];
                        }

                        // Calculate output index once for this (y, x, oc) combination
                        var outputIndex = y * _convolution.OutputWidth + x + 
                                          oc * _convolution.OutputWidth * _convolution.OutputHeight;
                        
                        if (_convolution.Activation == ActivationType.Sigmoid)
                            // Sigmoid activation function: 1 / (1 + e^(-x))
                            _buffers.Activations[activationOutputOffset + outputIndex] = 1.0f / (1.0f + float.Exp(-sum));
                        else if (_convolution.Activation == ActivationType.Relu)
                            // ReLU activation function: max(0, x)
                            _buffers.Activations[activationOutputOffset + outputIndex] = float.Max(0.0f, sum);
                        else if (_convolution.Activation == ActivationType.LeakyRelu)
                            // Leaky ReLU activation function: max(alpha * x, x)
                            _buffers.Activations[activationOutputOffset + outputIndex] = float.Max(0.1f * sum, sum);
                    }
                }
            }
        }
    });
}
  
    public void Backward()
    {
        Parallel.For(0, _buffers.BatchSize, batchIndex =>
        {
            // Loop over kernels per channel and image coordinates
            for (var kernelIndex = 0; kernelIndex < _convolution.KernelsPerChannel; kernelIndex++)
            {
                for (var y = 0; y < _convolution.OutputHeight; y++)
                {
                    for (var x = 0; x < _convolution.OutputWidth; x++)
                    {
                        for (var ic = 0; ic < _convolution.InputChannels; ic++)
                        {
                            var oc = ic * _convolution.KernelsPerChannel + kernelIndex;
                            var ic_pixel_offset = ic * _convolution.InputWidth * _convolution.InputHeight;

                            // Offsets for activations, errors, and gradients
                            var activationInputOffset = batchIndex * _networkData.ActivationCount + _convolution.ActivationInputOffset;
                            var currentErrorOffset = batchIndex * _networkData.ActivationCount + _convolution.CurrentLayerErrorOffset;
                            var nextErrorOffset = batchIndex * _networkData.ActivationCount + _convolution.NextLayerErrorOffset;
                            var gradientOffset = batchIndex * _networkData.ParameterCount + _convolution.ParameterOffset;

                            // Compute the output index for error propagation
                            var outputIndex = y * _convolution.OutputWidth + x + oc * _convolution.OutputWidth * _convolution.OutputHeight;

                            // Error from the next layer for this output position
                            var error = _buffers.Errors[nextErrorOffset + outputIndex];

                            var kernelSize = _convolution.KernelSize * _convolution.KernelSize;
                            // Start position for this kernel's weights in the parameter array
                            var kernelOffset = _convolution.ParameterOffset +
                                               oc * kernelSize;

                            // Accumulate gradients and propagate error to the input layer
                            for (var j = 0; j < kernelSize; j++)
                            {
                                var kernelY = j / _convolution.KernelSize;
                                var kernelX = j % _convolution.KernelSize;

                                var inputY = y * _convolution.Stride + kernelY - _convolution.Padding;
                                var inputX = x * _convolution.Stride + kernelX - _convolution.Padding;

                                if (inputY >= 0 && inputY < _convolution.InputHeight && inputX >= 0 && inputX < _convolution.InputWidth)
                                {
                                    var pixelIndex = ic_pixel_offset + inputY * _convolution.InputWidth + inputX;

                                    // Perform the backward pass logic with the adjusted pixelIndex
                                    var outputActivation = _buffers.Activations[activationInputOffset + pixelIndex];
                                    var derivative = 0.0f;
                                    if (_convolution.Activation == ActivationType.Sigmoid)
                                        // Sigmoid: Derivative of sigmoid is sigmoid(x) * (1 - sigmoid(x))
                                        derivative = outputActivation * (1.0f - outputActivation);
                                    else if (_convolution.Activation == ActivationType.Relu)
                                        // ReLU: Derivative of ReLU is 1 if x > 0, otherwise 0
                                        derivative = outputActivation > 0 ? 1.0f : 0.0f;
                                    else if (_convolution.Activation == ActivationType.LeakyRelu)
                                        // Leaky ReLU: Derivative of Leaky ReLU is alpha if x < 0, otherwise 1
                                        derivative = outputActivation > 0 ? 1.0f : 0.1f;

                                    var delta = error * derivative;

                                    // Accumulate the gradient
                                    _buffers.Gradients[gradientOffset + oc * kernelSize + j] =
                                        delta * _buffers.Activations[activationInputOffset + pixelIndex];

                                    // Propagate error to the current layer
                                    _buffers.Errors[currentErrorOffset + pixelIndex] +=
                                        delta * _buffers.Parameters[kernelOffset + j];
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    public bool IsTraining { get; set; }
}