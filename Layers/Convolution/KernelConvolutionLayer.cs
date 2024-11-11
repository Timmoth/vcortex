using ILGPU.Runtime;
using ILGPU;
using vcortex.Accelerated;

namespace vcortex.Layers.Convolution;

public class KernelConvolutionLayer : IConvolutionalLayer
{
    public KernelConvolutionLayer(int numKernels, int kernelSize = 3)
    {
        KernelSize = kernelSize;
        NumKernels = numKernels;
    }

    public int KernelSize { get; }
    public int NumKernels { get; }
    public int NumInputs => InputWidth * InputHeight * InputChannels;
    public int NumOutputs => OutputWidth * OutputHeight * OutputChannels;
    public int OutputWidth => InputWidth - KernelSize + 1;
    public int OutputHeight => InputHeight - KernelSize + 1;
    public int InputWidth { get; private set; }
    public int InputHeight { get; private set; }
    public int InputChannels { get; private set; }
    public int OutputChannels => NumKernels * InputChannels;
    public int GradientCount => OutputChannels * KernelSize * KernelSize;
    public int ActivationInputOffset { get; private set; }
    public int ActivationOutputOffset { get; private set; }
    public int CurrentLayerErrorOffset { get; private set; }
    public int NextLayerErrorOffset { get; private set; }
    public int GradientOffset { get; private set; }
    public int ParameterCount { get; private set; }
    public int ParameterOffset { get; private set; }
    public float[] Parameters { get; set; }

    public void Connect(IConvolutionalLayer prevLayer)
    {
        InputChannels = prevLayer.OutputChannels;
        InputWidth = prevLayer.OutputWidth;
        InputHeight = prevLayer.OutputHeight;

        ParameterCount = OutputChannels * KernelSize * KernelSize;
        ParameterOffset = prevLayer.ParameterOffset + prevLayer.ParameterCount;

        ActivationInputOffset = prevLayer.ActivationOutputOffset;
        ActivationOutputOffset = prevLayer.ActivationOutputOffset + prevLayer.NumOutputs;
        CurrentLayerErrorOffset = prevLayer.NextLayerErrorOffset;
        NextLayerErrorOffset = prevLayer.CurrentLayerErrorOffset + NumInputs;
        GradientOffset = prevLayer.GradientOffset + prevLayer.GradientCount;

    LayerData = new LayerData()
        {
            ActivationInputOffset = ActivationInputOffset,
            ActivationOutputOffset = ActivationOutputOffset,
            CurrentLayerErrorOffset = CurrentLayerErrorOffset,
            NextLayerErrorOffset = NextLayerErrorOffset,
            GradientOffset = GradientOffset,
            NumInputs = NumInputs,
            NumOutputs = NumOutputs,
            InputWidth = InputWidth,
            InputHeight = InputHeight,
            OutputWidth = OutputWidth,
            OutputHeight = OutputHeight,
            NumKernels = NumKernels,
            InputChannels = InputChannels,
            KernelSize = KernelSize,
            OutputChannels = OutputChannels,
        };
    }

    public void Connect(ConvolutionInputConfig config)
    {
        InputChannels = config.Grayscale ? 1 : 3;
        InputWidth = config.Width;
        InputHeight = config.Height;

        ParameterCount = OutputChannels * KernelSize * KernelSize;
        ParameterOffset = 0;

        ActivationInputOffset = 0;
        ActivationOutputOffset = NumInputs;
        CurrentLayerErrorOffset = 0;
        NextLayerErrorOffset = NumInputs;
        GradientOffset = 0;

        LayerData = new LayerData()
        {
            ActivationInputOffset = ActivationInputOffset,
            ActivationOutputOffset = ActivationOutputOffset,
            CurrentLayerErrorOffset = CurrentLayerErrorOffset,
            NextLayerErrorOffset = NextLayerErrorOffset,
            GradientOffset = GradientOffset,
            NumInputs = NumInputs,
            NumOutputs = NumOutputs,
            InputWidth = InputWidth,
            InputHeight = InputHeight,
            OutputWidth = OutputWidth,
            OutputHeight = OutputHeight,
            NumKernels = NumKernels,
            InputChannels = InputChannels,
            KernelSize = KernelSize,
            OutputChannels = OutputChannels,
        };
    }

    public void FillRandom()
    {

    }

    public virtual void FillRandom(NetworkAccelerator accelerator)
    {
        var parameters = new float[ParameterCount];

        var rnd = Random.Shared;
        var variance = 1.0f / (KernelSize * KernelSize * InputChannels);
        for (var i = 0; i < ParameterCount; i++)
        {
            var kernelOffset = ParameterOffset + KernelSize * KernelSize * i;
            for (var k = 0; k < KernelSize * KernelSize; k++)
                Parameters[kernelOffset + k] = (float)(rnd.NextDouble() * 2 - 1) * MathF.Sqrt(variance);
        }

        accelerator.Buffers.Parameters.View.SubView(LayerData.ParameterOffset, ParameterCount).CopyFromCPU(parameters);
    }

    public void Forward(float[] activations)
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

                        var kernelOffset = ParameterOffset + (k * InputChannels + ic) * KernelSize * KernelSize; // Kernel per input channel

                        for (var j = 0; j < KernelSize * KernelSize; j++)
                        {
                            var kernelY = j / KernelSize;
                            var kernelX = j % KernelSize;

                            var inputY = y + kernelY;
                            var inputX = x + kernelX;

                            var pixelIndex = inputY * InputWidth + inputX + ic * inputChannelOffset;
                            sum += activations[ActivationInputOffset + pixelIndex] * Parameters[kernelOffset + j];
                        }

                        activations[ActivationOutputOffset + outputIndex] = sum;
                    }
    }

    public void Backward(float[] activations, float[] errors,
        float[] gradients, float learningRate)
    {
        Array.Clear(errors, CurrentLayerErrorOffset, NumInputs);

        var inputChannelOffset = InputWidth * InputHeight;
        var outputChannelOffset = OutputWidth * OutputHeight;

        for (var k = 0; k < NumKernels; k++) // For each kernel (output channel)
            for (var y = 0; y < OutputHeight; y++)
                for (var x = 0; x < OutputWidth; x++)
                    for (var ic = 0; ic < InputChannels; ic++)
                    {
                        var outputIndex = y * OutputWidth + x + (k * InputChannels + ic) * outputChannelOffset;
                        var error = errors[NextLayerErrorOffset + outputIndex];

                        var kernelOffset = ParameterOffset + KernelSize * KernelSize * (k * InputChannels + ic); // Kernel per input channel

                        for (var j = 0; j < KernelSize * KernelSize; j++)
                        {
                            var kernelY = j / KernelSize;
                            var kernelX = j % KernelSize;

                            var inputY = y + kernelY;
                            var inputX = x + kernelX;

                            var pixelIndex = inputY * InputWidth + inputX + ic * inputChannelOffset;

                            // Accumulate the gradient for kernel updates
                            gradients[GradientOffset + (k * InputChannels + ic) * KernelSize * KernelSize + j] = error * activations[ActivationInputOffset + pixelIndex];

                            // Propagate the error to the current layer
                            errors[CurrentLayerErrorOffset + pixelIndex] += error * Parameters[kernelOffset + j];
                        }
                    }
    }


    public void AccumulateGradients(float[][] gradients, float learningRate)
    {
        var lr = learningRate / gradients.Length;

        for (var i = 0; i < OutputChannels; i++)
        {
            var kernelOffset = ParameterOffset + KernelSize * KernelSize * i;
            for (var j = 0; j < KernelSize * KernelSize; j++)
            {
                var gradientIndex = i * KernelSize + j;

                var kernelGradient = 0.0f;
                foreach (var gradient in gradients) kernelGradient += gradient[GradientOffset + gradientIndex];

                Parameters[kernelOffset + j] -= lr * kernelGradient;
            }
        }
    }

    public void Forward(NetworkAccelerator accelerator)
    {
        ForwardKernel(accelerator.Network.NetworkData.BatchSize * LayerData.NumKernels * LayerData.OutputHeight * LayerData.OutputWidth * LayerData.InputChannels, accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Parameters.View, accelerator.Buffers.Activations.View);
    }

    public void Backward(NetworkAccelerator accelerator)
    {
        for (int i = 0; i < accelerator.Network.NetworkData.BatchSize; i++)
        {
            accelerator.Buffers.Errors.View.SubView(accelerator.Network.NetworkData.ErrorCount * i + LayerData.CurrentLayerErrorOffset, LayerData.NumInputs).MemSetToZero();
        }

        BackwardKernel(accelerator.Network.NetworkData.BatchSize * LayerData.NumKernels * LayerData.OutputHeight * LayerData.OutputWidth * LayerData.InputChannels, accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Parameters.View, accelerator.Buffers.Activations.View, accelerator.Buffers.Gradients.View,
            accelerator.Buffers.Errors.View);
    }

    public void AccumulateGradients(NetworkAccelerator accelerator)
    {
        GradientAccumulationKernel(LayerData.OutputChannels, accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Parameters.View, accelerator.Buffers.Gradients.View);
    }

    public static void ForwardKernelImpl(
    Index1D index,
    NetworkData networkData,
    LayerData layerData,
    ArrayView<float> parameters,
    ArrayView<float> activations)
    {
        // index = batches * NumKernels * height * width * InputChannels
        int batch = index / (layerData.NumKernels * layerData.OutputHeight * layerData.OutputWidth * layerData.InputChannels);
        int k = (index / (layerData.OutputHeight * layerData.OutputWidth * layerData.InputChannels)) % layerData.NumKernels;
        int y = (index / (layerData.OutputWidth * layerData.InputChannels)) % layerData.OutputHeight;
        int x = (index / layerData.InputChannels) % layerData.OutputWidth;
        int ic = index % layerData.InputChannels;

        var activationInputOffset = batch * networkData.ActivationCount + layerData.ActivationInputOffset;
        var activationOutputOffset = batch * networkData.ActivationCount + layerData.ActivationOutputOffset;

        var inputChannelOffset = layerData.InputWidth * layerData.InputHeight;
        var outputChannelOffset = layerData.OutputWidth * layerData.OutputHeight;

        var outputIndex = y * layerData.OutputWidth + x + (k * layerData.InputChannels + ic) * outputChannelOffset;
        float sum = 0;

        var kernelOffset = layerData.ParameterOffset + (k * layerData.InputChannels + ic) * layerData.KernelSize * layerData.KernelSize; // Kernel per input channel

        for (var j = 0; j < layerData.KernelSize * layerData.KernelSize; j++)
        {
            var kernelY = j / layerData.KernelSize;
            var kernelX = j % layerData.KernelSize;

            var inputY = y + kernelY;
            var inputX = x + kernelX;

            var pixelIndex = inputY * layerData.InputWidth + inputX + ic * inputChannelOffset;
            sum += activations[activationInputOffset + pixelIndex] * parameters[kernelOffset + j];
        }

        activations[activationOutputOffset + outputIndex] = sum;
    }

    public static void BackwardKernelImpl(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> parameters,
        ArrayView<float> activations,
        ArrayView<float> gradients,
        ArrayView<float> errors)
    {
        // index = batches * NumKernels * height * width * InputChannels
        int batch = index / (layerData.NumKernels * layerData.OutputHeight * layerData.OutputWidth * layerData.InputChannels);
        int k = (index / (layerData.OutputHeight * layerData.OutputWidth * layerData.InputChannels)) % layerData.NumKernels;
        int y = (index / (layerData.OutputWidth * layerData.InputChannels)) % layerData.OutputHeight;
        int x = (index / layerData.InputChannels) % layerData.OutputWidth;
        int ic = index % layerData.InputChannels;

        var activationInputOffset = batch * networkData.ActivationCount + layerData.ActivationInputOffset;
        var currentErrorOffset = batch * networkData.ErrorCount + layerData.CurrentLayerErrorOffset;
        var nextErrorOffset = batch * networkData.ErrorCount + layerData.NextLayerErrorOffset;
        var gradientOffset = batch * networkData.GradientCount + layerData.GradientOffset;

        var inputChannelOffset = layerData.InputWidth * layerData.InputHeight;
        var outputChannelOffset = layerData.OutputWidth * layerData.OutputHeight;

        var outputIndex = y * layerData.OutputWidth + x + (k * layerData.InputChannels + ic) * outputChannelOffset;
        var error = errors[nextErrorOffset + outputIndex];

        var kernelOffset = layerData.ParameterOffset + layerData.KernelSize * layerData.KernelSize * (k * layerData.InputChannels + ic); // Kernel per input channel

        for (var j = 0; j < layerData.KernelSize * layerData.KernelSize; j++)
        {
            var kernelY = j / layerData.KernelSize;
            var kernelX = j % layerData.KernelSize;

            var inputY = y + kernelY;
            var inputX = x + kernelX;

            var pixelIndex = inputY * layerData.InputWidth + inputX + ic * inputChannelOffset;

            // Accumulate the gradient for kernel updates
            gradients[gradientOffset + (k * layerData.InputChannels + ic) * layerData.KernelSize * layerData.KernelSize + j] = error * activations[activationInputOffset + pixelIndex];

            // Propagate the error to the current layer
            errors[currentErrorOffset + pixelIndex] += error * parameters[kernelOffset + j];
        }
    }
    public static void GradientAccumulationKernelImpl(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> parameters,
        ArrayView<float> gradients)
    {
        // Number of samples in the batch
        var batchSize = networkData.BatchSize;
        var lr = networkData.LearningRate / batchSize; // Scale learning rate by batch size for averaging

        var kernelOffset = layerData.ParameterOffset + layerData.KernelSize * layerData.KernelSize * index;
        for (var j = 0; j < layerData.KernelSize * layerData.KernelSize; j++)
        {
            var gradientIndex = index * layerData.KernelSize + j;

            var kernelGradient = 0.0f;
            for (var i = 0; i < networkData.BatchSize; i++)
            {
                kernelGradient += gradients[i * networkData.GradientCount + layerData.GradientOffset + gradientIndex];
            }

            parameters[kernelOffset + j] -= lr * kernelGradient;
        }
    }

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>> ForwardKernel { get; private set; }

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> BackwardKernel
    { get; private set; }

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>> GradientAccumulationKernel { get; private set; }


    public void CompileKernels(Accelerator accelerator)
    {
        ForwardKernel =
            accelerator.LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>>(
                ForwardKernelImpl);
        BackwardKernel =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(BackwardKernelImpl);
        GradientAccumulationKernel =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>>(GradientAccumulationKernelImpl);
    }

    public LayerData LayerData { get; private set; }
}