namespace vcortex.Layers.Convolution;

public class MaxPoolingConvolutionLayer : IConvolutionalLayer
{
    public MaxPoolingConvolutionLayer(int poolSize)
    {
        PoolSize = poolSize;
    }

    public int PoolSize { get; }
    public int OutputWidth => InputWidth / PoolSize;
    public int OutputHeight => InputHeight / PoolSize;

    public int NumInputs => InputWidth * InputHeight * InputChannels;
    public int NumOutputs => OutputWidth * OutputHeight * InputChannels;

    public int InputWidth { get; private set; }
    public int InputHeight { get; private set; }
    public int InputChannels { get; private set; }
    public int OutputChannels { get; private set; }

    public void Connect(IConvolutionalLayer prevLayer)
    {
        InputWidth = prevLayer.OutputWidth;
        InputHeight = prevLayer.OutputHeight;
        OutputChannels = InputChannels = prevLayer.OutputChannels;
    }

    public void Connect(ConvolutionInputConfig config)
    {
        InputWidth = config.Width;
        InputHeight = config.Height;
        OutputChannels = InputChannels = config.Grayscale ? 1 : 3;
    }

    public void Forward(float[] inputs, float[] outputs)
    {
        var inputChannelOffset = InputWidth * InputHeight;
        var outputChannelOffset = OutputWidth * OutputHeight;

        for (var c = 0; c < InputChannels; c++)
        for (var y = 0; y < OutputHeight; y++)
        for (var x = 0; x < OutputWidth; x++)
        {
            var outputIndex = y * OutputWidth + x + c * outputChannelOffset;

            var max = float.MinValue;

            // SIMD for pooling region, process multiple elements at once
            for (var ky = 0; ky < PoolSize; ky++)
            for (var kx = 0; kx < PoolSize; kx++)
            {
                var oldX = x * PoolSize + kx;
                var oldY = y * PoolSize + ky;
                var inputIndex = oldY * InputWidth + oldX + c * inputChannelOffset;

                // Here, SIMD could help if PoolSize is large, by comparing multiple input elements at once
                max = Math.Max(max, inputs[inputIndex]);
            }

            outputs[outputIndex] = max;
        }
    }

    public int GradientCount => 0;

    public void Backward(float[] inputs, float[] outputs, float[] currentLayerErrors, float[] nextLayerErrors,
        float[] gradients,
        float learningRate)
    {
        Array.Clear(currentLayerErrors, 0, currentLayerErrors.Length);
        var inputChannelOffset = InputWidth * InputHeight;
        var outputChannelOffset = OutputWidth * OutputHeight;

        for (var c = 0; c < InputChannels; c++)
        for (var y = 0; y < OutputHeight; y++)
        for (var x = 0; x < OutputWidth; x++)
        {
            var outputIndex = y * OutputWidth + x + c * outputChannelOffset;

            var max = float.MinValue;
            var maxIndex = -1;

            // Find max position within pool window
            for (var ky = 0; ky < PoolSize; ky++)
            for (var kx = 0; kx < PoolSize; kx++)
            {
                var oldX = x * PoolSize + kx;
                var oldY = y * PoolSize + ky;
                var inputIndex = oldY * InputWidth + oldX + c * inputChannelOffset;

                if (inputs[inputIndex] > max)
                {
                    max = inputs[inputIndex];
                    maxIndex = inputIndex;
                }
            }

            if (maxIndex >= 0)
                // Propagate error to the max position only
                currentLayerErrors[maxIndex] = nextLayerErrors[outputIndex];
        }
    }

    public void AccumulateGradients(float[][] gradients, float learningRate)
    {
    }
}