using ILGPU;
using ILGPU.Runtime;
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

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>> ForwardKernel
    {
        get;
        private set;
    }

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> BackwardKernel { get; private set; }

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> GradientAccumulationKernel { get; private set; }

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
    public LayerData LayerData { get; set; }
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


        LayerData = new LayerData(NumInputs, NumOutputs, ActivationInputOffset, ActivationOutputOffset, GradientOffset,
            NextLayerErrorOffset, CurrentLayerErrorOffset, ParameterOffset, 0, InputWidth, InputHeight, OutputWidth,
            OutputHeight, InputChannels, OutputChannels, NumKernels, KernelSize, 0);
    }
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

        LayerData = new LayerData(NumInputs, NumOutputs, ActivationInputOffset, ActivationOutputOffset, GradientOffset,
            NextLayerErrorOffset, CurrentLayerErrorOffset, ParameterOffset, 0, InputWidth, InputHeight, OutputWidth,
            OutputHeight, InputChannels, OutputChannels, NumKernels, KernelSize, 0);
    }
    
    public virtual void FillRandom(NetworkAccelerator accelerator)
    {
        var parameters = new float[ParameterCount];

        var rnd = Random.Shared;
        var variance = 2.0f / (ParameterCount);
        for (var i = 0; i < ParameterCount; i++)
        {
            parameters[i] = (float)(rnd.NextDouble() * 2 - 1) * MathF.Sqrt(variance);
        }

        accelerator.Buffers.Parameters.View.SubView(LayerData.ParameterOffset, ParameterCount).CopyFromCPU(parameters);
    }

    public void Forward(NetworkAccelerator accelerator)
    {
        ForwardKernel(
            accelerator.Network.NetworkData.BatchSize * LayerData.NumKernels * LayerData.OutputHeight *
            LayerData.OutputWidth * LayerData.InputChannels, accelerator.Network.NetworkData, LayerData,
            accelerator.Buffers.Parameters.View, accelerator.Buffers.Activations.View);
    }

    public void Backward(NetworkAccelerator accelerator)
    {
        BackwardKernel(
            accelerator.Network.NetworkData.BatchSize * LayerData.NumKernels * LayerData.OutputHeight *
            LayerData.OutputWidth * LayerData.InputChannels, accelerator.Network.NetworkData, LayerData,
            accelerator.Buffers.Parameters.View, accelerator.Buffers.Activations.View,
            accelerator.Buffers.Gradients.View,
            accelerator.Buffers.Errors.View);
    }

    public void AccumulateGradients(NetworkAccelerator accelerator)
    {
        GradientAccumulationKernel(LayerData.OutputChannels * LayerData.KernelSize * LayerData.KernelSize,
            accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Parameters.View,
            accelerator.Buffers.Gradients.View, accelerator.Buffers.FirstMoment.View,
            accelerator.Buffers.SecondMoment.View);
    }

    #region Kernels
    public void CompileKernels(Accelerator accelerator)
    {
        ForwardKernel =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>>(
                    ForwardKernelImpl);
        BackwardKernel =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(BackwardKernelImpl);
        GradientAccumulationKernel =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(GradientAccumulationKernelImpl);
    }


    public static void ForwardKernelImpl(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> parameters,
        ArrayView<float> activations)
    {
        var batch = index / (layerData.NumKernels * layerData.OutputHeight * layerData.OutputWidth *
                             layerData.InputChannels);
        var k = index / (layerData.OutputHeight * layerData.OutputWidth * layerData.InputChannels) %
                layerData.NumKernels;
        var y = index / (layerData.OutputWidth * layerData.InputChannels) % layerData.OutputHeight;
        var x = index / layerData.InputChannels % layerData.OutputWidth;
        var ic = index % layerData.InputChannels;
        var oc = ic * layerData.NumKernels + k;
        var ic_pixel_offset = ic * layerData.InputWidth * layerData.InputHeight;
        
        // Offsets in the input/output activations
        var activationInputOffset = batch * networkData.ActivationCount + layerData.ActivationInputOffset;
        var activationOutputOffset = batch * networkData.ActivationCount + layerData.ActivationOutputOffset;
        var kernelSize = layerData.KernelSize * layerData.KernelSize;

        // Kernel offset to access weights for this specific kernel and input channel
        var kernelOffset = layerData.ParameterOffset +
                           oc * kernelSize;

        // Accumulate the weighted sum of inputs within the convolutional kernel window
        float sum = 0;
        for (var j = 0; j < kernelSize; j++)
        {
            // Convert linear kernel index to 2D (kernelY, kernelX) coordinates
            var kernelY = j / layerData.KernelSize;
            var kernelX = j % layerData.KernelSize;

            // Calculate corresponding input coordinates for (kernelY, kernelX)
            var inputY = y + kernelY;
            var inputX = x + kernelX;

            var pixelIndex = ic_pixel_offset + inputY * layerData.InputWidth + inputX;
            sum += activations[activationInputOffset + pixelIndex] * parameters[kernelOffset + j];
        }

        // Calculate total output position for this batch, kernel, height, width, and channel
        var outputIndex = y * layerData.OutputWidth + x + oc * layerData.OutputWidth * layerData.OutputHeight;
        
        // Store the result in the output activations
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
        // Calculate batch, kernel, output height (y), width (x), and input channel
        int batch = index / (layerData.NumKernels * layerData.OutputHeight * layerData.OutputWidth *
                             layerData.InputChannels);
        var k = index / (layerData.OutputHeight * layerData.OutputWidth * layerData.InputChannels) %
                layerData.NumKernels;
        var y = index / (layerData.OutputWidth * layerData.InputChannels) % layerData.OutputHeight;
        var x = index / layerData.InputChannels % layerData.OutputWidth;
        var ic = index % layerData.InputChannels;
        var oc = ic * layerData.NumKernels + k;
        var ic_pixel_offset = ic * layerData.InputWidth * layerData.InputHeight;

        // Offsets for activations, errors, and gradients
        var activationInputOffset = batch * networkData.ActivationCount + layerData.ActivationInputOffset;
        var currentErrorOffset = batch * networkData.ErrorCount + layerData.CurrentLayerErrorOffset;
        var nextErrorOffset = batch * networkData.ErrorCount + layerData.NextLayerErrorOffset;
        var gradientOffset = batch * networkData.GradientCount + layerData.GradientOffset;

        // Compute the output index for error propagation
        var outputIndex = y * layerData.OutputWidth + x + oc * layerData.OutputWidth * layerData.OutputHeight;

        // Error from the next layer for this output position
        var error = errors[nextErrorOffset + outputIndex];

        var kernelSize = layerData.KernelSize * layerData.KernelSize;
        // Start position for this kernel's weights in the parameter array
        var kernelOffset = layerData.ParameterOffset +
                           oc * kernelSize;

        // Accumulate gradients and propagate error to the input layer
        for (var j = 0; j < kernelSize; j++)
        {
            var kernelY = j / layerData.KernelSize;
            var kernelX = j % layerData.KernelSize;

            var inputY = y + kernelY;
            var inputX = x + kernelX;

            var pixelIndex = ic_pixel_offset + inputY * layerData.InputWidth + inputX;

            // Accumulate the gradient
            gradients[
                gradientOffset + oc * kernelSize +
                j] = error * activations[activationInputOffset + pixelIndex];

            // Propagate error to the current layer
            Atomic.Add(ref errors[currentErrorOffset + pixelIndex], error * parameters[kernelOffset + j]);
        }
    }

    public static void GradientAccumulationKernelImpl(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> parameters,
        ArrayView<float> gradients,
        ArrayView<float> firstMoment,
        ArrayView<float> secondMoment)
    {
        // Compute the kernel size, batch size, and indices for channel and kernel
        var kernelSize = layerData.KernelSize * layerData.KernelSize;
        var batchSize = networkData.BatchSize;
        var oc = index / kernelSize;
        var kernelIndex = index % kernelSize;

        // Calculate parameter and gradient indices
        var kernelOffset = layerData.ParameterOffset + kernelSize * oc + kernelIndex;
        var gradientIndex = layerData.GradientOffset + oc * kernelSize + kernelIndex;

        // Accumulate gradients across the batch for each parameter
        var kernelGradient = 0.0f;
        for (var i = 0; i < batchSize; i++) kernelGradient += gradients[i * networkData.GradientCount + gradientIndex];

        // Average the gradient over the batch size
        kernelGradient /= batchSize;

        // Update the first and second moment estimates for the Adam optimizer
        firstMoment[gradientIndex] =
            networkData.Beta1 * firstMoment[gradientIndex] + (1 - networkData.Beta1) * kernelGradient;
        secondMoment[gradientIndex] = networkData.Beta2 * secondMoment[gradientIndex] +
                                      (1 - networkData.Beta2) * kernelGradient * kernelGradient;

        // Apply bias correction for the moments
        var mHat = firstMoment[gradientIndex] / (1 - MathF.Pow(networkData.Beta1, networkData.Timestep));
        var vHat = secondMoment[gradientIndex] / (1 - MathF.Pow(networkData.Beta2, networkData.Timestep));

        // Update parameter using Adam optimizer formula
        parameters[kernelOffset] -=
            networkData.LearningRate * mHat / (MathF.Sqrt(vHat) + networkData.Epsilon);
    }
    
    #endregion

}