using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using vcortex.Accelerated;

namespace vcortex.Layers.Connected;

public class SoftmaxConnectedLayer : IConnectedLayer
{
    public SoftmaxConnectedLayer(int numOutputs)
    {
        NumOutputs = numOutputs;
    }

    public int BiasOffset => NumInputs * NumOutputs;

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>> ForwardKernel1
    {
        get;
        private set;
    }

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>> ForwardKernel2
    {
        get;
        private set;
    }

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> BackwardKernel1 { get; private set; }

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> BackwardKernel2 { get; private set; }

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> GradientAccumulationKernel { get; private set; }

    public int NumInputs { get; private set; }
    public int NumOutputs { get; }
    public int GradientCount => NumInputs * NumOutputs + NumOutputs;
    public int ActivationInputOffset { get; private set; }
    public int ActivationOutputOffset { get; private set; }
    public int CurrentLayerErrorOffset { get; private set; }
    public int NextLayerErrorOffset { get; private set; }
    public int GradientOffset { get; private set; }
    public int ParameterCount { get; private set; }
    public int ParameterOffset { get; private set; }

    public void Connect(ILayer prevLayer)
    {
        NumInputs = prevLayer.NumOutputs;

        ParameterCount = NumOutputs * NumInputs + NumOutputs;
        ParameterOffset = prevLayer.ParameterOffset + prevLayer.ParameterCount;

        ActivationInputOffset = prevLayer.ActivationOutputOffset;
        ActivationOutputOffset = prevLayer.ActivationOutputOffset + prevLayer.NumOutputs;
        CurrentLayerErrorOffset = prevLayer.NextLayerErrorOffset;
        NextLayerErrorOffset = prevLayer.CurrentLayerErrorOffset + NumInputs;
        GradientOffset = prevLayer.GradientOffset + prevLayer.GradientCount;

        LayerData = new LayerData(NumInputs, NumOutputs, ActivationInputOffset, ActivationOutputOffset, GradientOffset,
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
        GradientOffset = 0;

        LayerData = new LayerData(NumInputs, NumOutputs, ActivationInputOffset, ActivationOutputOffset, GradientOffset,
            NextLayerErrorOffset, CurrentLayerErrorOffset, ParameterOffset, BiasOffset, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    }

    public virtual void FillRandom(NetworkAccelerator accelerator)
    {
        var parameters = new float[ParameterCount];

        var rnd = Random.Shared;
        var variance = 1.0f / (ParameterCount);
        for (var i = 0; i < ParameterCount; i++)
        {
            parameters[i] = (float)(rnd.NextDouble() * 2 - 1) * MathF.Sqrt(variance);
        }

        accelerator.Buffers.Parameters.View.SubView(LayerData.ParameterOffset, ParameterCount).CopyFromCPU(parameters);
    }


    public void Forward(NetworkAccelerator accelerator)
    {
        ForwardKernel1(LayerData.NumOutputs * accelerator.Network.NetworkData.BatchSize,
            accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Parameters.View,
            accelerator.Buffers.Activations.View);
        accelerator.accelerator.Synchronize();

        ForwardKernel2(accelerator.Network.NetworkData.BatchSize, accelerator.Network.NetworkData, LayerData,
            accelerator.Buffers.Parameters.View, accelerator.Buffers.Activations.View);
    }

    public void Backward(NetworkAccelerator accelerator)
    {
        BackwardKernel1(LayerData.NumOutputs * LayerData.NumInputs * accelerator.Network.NetworkData.BatchSize,
            accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Parameters.View,
            accelerator.Buffers.Activations.View, accelerator.Buffers.Gradients.View,
            accelerator.Buffers.Errors.View);
        accelerator.accelerator.Synchronize();

        BackwardKernel2(LayerData.NumOutputs * accelerator.Network.NetworkData.BatchSize,
            accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Parameters.View,
            accelerator.Buffers.Activations.View, accelerator.Buffers.Gradients.View,
            accelerator.Buffers.Errors.View);
    }

    public void AccumulateGradients(NetworkAccelerator accelerator)
    {
        GradientAccumulationKernel(LayerData.NumOutputs, accelerator.Network.NetworkData, LayerData,
            accelerator.Buffers.Parameters.View, accelerator.Buffers.Gradients.View,
            accelerator.Buffers.FirstMoment.View, accelerator.Buffers.SecondMoment.View);
    }

    public LayerData LayerData { get; set; }


    public void CompileKernels(NetworkAccelerator accelerator)
    {
        ForwardKernel1 =
            accelerator.accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>>(
                    ForwardKernel1Impl);

        ForwardKernel2 =
            accelerator.accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>>(
                    ForwardKernel2Impl);

        BackwardKernel1 =
            accelerator.accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(BackwardKernel1Impl);

        BackwardKernel2 =
            accelerator.accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(BackwardKernel2Impl);

        GradientAccumulationKernel =
            accelerator.accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(GradientAccumulationKernelImpl);
    }

    public static void ForwardKernel1Impl(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> parameters,
        ArrayView<float> activations)
    {
        // Calculate output index and batch offset
        var batchIndex = index / layerData.NumOutputs;
        var outputIndex = index % layerData.NumOutputs;

        // Activation offsets
        var activationInputOffset = batchIndex * networkData.ActivationCount + layerData.ActivationInputOffset;
        var activationOutputOffset = batchIndex * networkData.ActivationCount + layerData.ActivationOutputOffset;

        // Initialize the sum with the bias term
        var sum = parameters[layerData.ParameterOffset + layerData.BiasOffset + outputIndex];

        // Offset for weights for this output neuron
        var weightsOffset = layerData.ParameterOffset + layerData.NumInputs * outputIndex;

        // Compute the weighted sum for this output
        for (var j = 0; j < layerData.NumInputs; j++)
            sum += activations[activationInputOffset + j] * parameters[weightsOffset + j];

        // Store the result as the "raw score" in the output activations
        activations[activationOutputOffset + outputIndex] = sum;
    }

    public static void ForwardKernel2Impl(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> parameters,
        ArrayView<float> activations)
    {
        var batchOffset = index;
        var activationOutputOffset = batchOffset * networkData.ActivationCount + layerData.ActivationOutputOffset;

        // Find the maximum value in the outputs for numerical stability
        var maxVal = activations[activationOutputOffset];
        for (var i = 1; i < layerData.NumOutputs; i++)
            maxVal = XMath.Max(maxVal, activations[activationOutputOffset + i]);

        // Compute the sum of exponentials
        float sumExp = 0;
        for (var i = 0; i < layerData.NumOutputs; i++)
        {
            activations[activationOutputOffset + i] = XMath.Exp(activations[activationOutputOffset + i] - maxVal);
            sumExp += activations[activationOutputOffset + i];
        }

        // Normalize each output to get softmax probabilities
        // for (var i = 0; i < layerData.NumOutputs; i++)
        //     activations[activationOutputOffset + i] /= sumExp;
        
        if (sumExp <= 0 || float.IsNaN(sumExp) || float.IsInfinity(sumExp))
        {
            var uniformProb = 1.0f / layerData.NumOutputs;
            for (var i = 0; i < layerData.NumOutputs; i++) activations[activationOutputOffset + i] = uniformProb;
        }
        else
        {
            // Normalize to get probabilities
            for (var i = 0; i < layerData.NumOutputs; i++)
                activations[activationOutputOffset + i] /= sumExp;
        }
    }

    public static void BackwardKernel1Impl(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> parameters,
        ArrayView<float> activations,
        ArrayView<float> gradients,
        ArrayView<float> errors)
    {
        int batchIndex = index / (layerData.NumInputs * layerData.NumOutputs);
        var outputIndex = index / layerData.NumInputs % layerData.NumOutputs;
        var inputIndex = index % layerData.NumInputs;

        // Offsets for activations and errors
        var activationInputOffset = batchIndex * networkData.ActivationCount + layerData.ActivationInputOffset;
        var currentErrorOffset = batchIndex * networkData.ErrorCount + layerData.CurrentLayerErrorOffset;
        var nextErrorOffset = batchIndex * networkData.ErrorCount + layerData.NextLayerErrorOffset;
        var gradientOffset = batchIndex * networkData.GradientCount + layerData.GradientOffset;

        // Calculate delta for this output neuron
        var delta = errors[nextErrorOffset + outputIndex];

        // Compute the gradient contribution for each weight and accumulate errors for backpropagation
        Atomic.Add(ref errors[currentErrorOffset + inputIndex], delta * parameters[layerData.ParameterOffset + layerData.NumInputs * outputIndex + inputIndex]);

        // Store gradient for the current weight
        gradients[gradientOffset + outputIndex * layerData.NumInputs + inputIndex] =
            delta * activations[activationInputOffset + inputIndex];
    }

    public static void BackwardKernel2Impl(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> parameters,
        ArrayView<float> activations,
        ArrayView<float> gradients,
        ArrayView<float> errors)
    {
        var batchIndex = index / layerData.NumOutputs;
        var outputIndex = index % layerData.NumOutputs;

        var nextErrorOffset = batchIndex * networkData.ErrorCount + layerData.NextLayerErrorOffset;
        var gradientOffset = batchIndex * networkData.GradientCount + layerData.GradientOffset;

        // Calculate gradient for the bias term of this output neuron
        var delta = errors[nextErrorOffset + outputIndex];
        gradients[gradientOffset + layerData.BiasOffset + outputIndex] = delta;
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
        var batchSize = networkData.BatchSize;
        var outputIndex = index;

        var weightOffset = layerData.ParameterOffset + layerData.NumInputs * outputIndex;

        // Accumulate gradients across the batch
        for (var j = 0; j < layerData.NumInputs; j++)
        {
            var gradientIndex = layerData.GradientOffset + outputIndex * layerData.NumInputs + j;

            // Mean gradient over batch
            var weightGradient = 0.0f;
            for (var i = 0; i < batchSize; i++)
                weightGradient += gradients[networkData.GradientCount * i + gradientIndex];

            weightGradient /= batchSize;

            // Update first and second moments
            firstMoment[gradientIndex] = networkData.Beta1 * firstMoment[gradientIndex] +
                                         (1 - networkData.Beta1) * weightGradient;
            secondMoment[gradientIndex] = networkData.Beta2 * secondMoment[gradientIndex] +
                                          (1 - networkData.Beta2) * weightGradient * weightGradient;

            // Bias correction and parameter update
            var mHat = firstMoment[gradientIndex] / (1 - MathF.Pow(networkData.Beta1, networkData.Timestep));
            var vHat = secondMoment[gradientIndex] / (1 - MathF.Pow(networkData.Beta2, networkData.Timestep));
            parameters[weightOffset + j] -= networkData.LearningRate * mHat / (MathF.Sqrt(vHat) + networkData.Epsilon);
        }

        // Handle bias gradients separately
        var gradientIndex2 = layerData.GradientOffset + layerData.BiasOffset + outputIndex;
        var biasGradient = 0.0f;
        for (var i = 0; i < batchSize; i++)
            biasGradient += gradients[networkData.GradientCount * i + gradientIndex2];

        biasGradient /= batchSize;

        firstMoment[gradientIndex2] =
            networkData.Beta1 * firstMoment[gradientIndex2] + (1 - networkData.Beta1) * biasGradient;
        secondMoment[gradientIndex2] = networkData.Beta2 * secondMoment[gradientIndex2] +
                                       (1 - networkData.Beta2) * biasGradient * biasGradient;

        var mHat2 = firstMoment[gradientIndex2] / (1 - MathF.Pow(networkData.Beta1, networkData.Timestep));
        var vHat2 = secondMoment[gradientIndex2] / (1 - MathF.Pow(networkData.Beta2, networkData.Timestep));
        parameters[layerData.ParameterOffset + layerData.BiasOffset + outputIndex] -=
            networkData.LearningRate * mHat2 / (MathF.Sqrt(vHat2) + networkData.Epsilon);
    }
}