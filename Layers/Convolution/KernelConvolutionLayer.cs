namespace vcortex.Layers.Convolution;

public class KernelConvolutionLayer : IConvolutionalLayer
{
    public KernelConvolutionLayer(int numKernels, int kernelSize = 3)
    {
        KernelSize = kernelSize;
        NumKernels = numKernels;
    }

    public float[][] Kernels { get; private set; }
    public int KernelSize { get; }
    public int NumKernels { get; }
    public int NumInputs => InputWidth * InputHeight * InputChannels;
    public int NumOutputs => OutputWidth * OutputHeight * OutputChannels;

    public int InputWidth { get; private set; }
    public int InputHeight { get; private set; }
    public int InputChannels { get; private set; }
    public int OutputChannels => NumKernels * InputChannels;
    public int GradientCount => OutputChannels * KernelSize * KernelSize;

    public void Connect(IConvolutionalLayer prevLayer)
    {
        InputChannels = prevLayer.OutputChannels;
        InputWidth = prevLayer.OutputWidth;
        InputHeight = prevLayer.OutputHeight;
        Kernels = new float[OutputChannels][];

        for (var i = 0; i < OutputChannels; i++) Kernels[i] = new float[KernelSize * KernelSize];
    }

    public void Connect(ConvolutionInputConfig config)
    {
        InputChannels = config.Grayscale ? 1 : 3;
        InputWidth = config.Width;
        InputHeight = config.Height;
        Kernels = new float[OutputChannels][];

        for (var i = 0; i < OutputChannels; i++) Kernels[i] = new float[KernelSize * KernelSize];
    }

    public int OutputWidth => InputWidth - KernelSize + 1;
    public int OutputHeight => InputHeight - KernelSize + 1;

    public void FillRandom()
    {
        var rnd = Random.Shared;
        var variance = 1.0f / (KernelSize * KernelSize * InputChannels);
        for (var i = 0; i < Kernels.Length; i++)
        {
            var kernel = Kernels[i];
            for (var k = 0; k < KernelSize * KernelSize; k++)
                kernel[k] = (float)(rnd.NextDouble() * 2 - 1) * MathF.Sqrt(variance);
        }
    }

    public void Forward(float[] inputs, float[] outputs)
    {
        var inputChannelOffset = InputWidth * InputHeight;
        var outputChannelOffset = OutputWidth * OutputHeight;

        for (var k = 0; k < NumKernels; k++) // Iterate over each kernel (output channel)
        for (var y = 0; y < OutputHeight; y++)
        for (var x = 0; x < OutputWidth; x++)
            // Accumulate convolutions across all input channels e.g (R, G, B)
        for (var ic = 0; ic < InputChannels; ic++)
        {
            var outputIndex = y * OutputWidth + x + (k * InputChannels + ic) * outputChannelOffset;
            float sum = 0;

            var kernel = Kernels[k * InputChannels + ic]; // Kernel per input channel

            for (var j = 0; j < KernelSize * KernelSize; j++)
            {
                var kernelY = j / KernelSize;
                var kernelX = j % KernelSize;

                var inputY = y + kernelY;
                var inputX = x + kernelX;

                var pixelIndex = inputY * InputWidth + inputX + ic * inputChannelOffset;
                sum += inputs[pixelIndex] * kernel[j];
            }

            outputs[outputIndex] = sum;
        }
    }

    public void Backward(float[] inputs, float[] outputs, float[] currentLayerErrors, float[] nextLayerErrors,
        float[] gradients, float learningRate)
    {
        Array.Clear(currentLayerErrors, 0, currentLayerErrors.Length);

        var inputChannelOffset = InputWidth * InputHeight;
        var outputChannelOffset = OutputWidth * OutputHeight;

        for (var k = 0; k < NumKernels; k++) // For each kernel (output channel)
        for (var y = 0; y < OutputHeight; y++)
        for (var x = 0; x < OutputWidth; x++)
        for (var ic = 0; ic < InputChannels; ic++)
        {
            var outputIndex = y * OutputWidth + x + (k * InputChannels + ic) * outputChannelOffset;
            var error = nextLayerErrors[outputIndex];

            var kernel = Kernels[k * InputChannels + ic]; // Kernel per input channel

            for (var j = 0; j < KernelSize * KernelSize; j++)
            {
                var kernelY = j / KernelSize;
                var kernelX = j % KernelSize;

                var inputY = y + kernelY;
                var inputX = x + kernelX;

                var pixelIndex = inputY * InputWidth + inputX + ic * inputChannelOffset;

                // Accumulate the gradient for kernel updates
                gradients[(k * InputChannels + ic) * KernelSize * KernelSize + j] = error * inputs[pixelIndex];

                // Propagate the error to the current layer
                currentLayerErrors[pixelIndex] += error * kernel[j];
            }
        }
    }


    public void AccumulateGradients(float[][] gradients, float learningRate)
    {
        var lr = learningRate / gradients.Length;

        for (var i = 0; i < OutputChannels; i++)
        {
            var kernel = Kernels[i];
            for (var j = 0; j < KernelSize * KernelSize; j++)
            {
                var gradientIndex = i * KernelSize + j;

                var kernelGradient = 0.0f;
                foreach (var gradient in gradients) kernelGradient += gradient[gradientIndex];

                kernel[j] -= lr * kernelGradient;
            }
        }
    }
}