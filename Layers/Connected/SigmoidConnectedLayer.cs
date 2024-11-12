using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using vcortex.Accelerated;

namespace vcortex.Layers.Connected;

public class SigmoidConnectedLayer : IConnectedLayer
{
    public SigmoidConnectedLayer(int numOutputs)
    {
        NumOutputs = numOutputs;
    }

    public int BiasOffset => NumInputs * NumOutputs;

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>> ForwardKernel
    {
        get;
        private set;
    }

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> BackwardKernel1 { get; private set; }

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> BackwardKernel2 { get; private set; }

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> GradientAccumulationKernel1 { get; private set; }

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> GradientAccumulationKernel2 { get; private set; }

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
        var rnd = Random.Shared;
        var limit = MathF.Sqrt(6.0f / (NumInputs + NumOutputs));

        var parameters = new float[ParameterCount];

        for (var i = 0; i < NumOutputs; i++)
        {
            parameters[BiasOffset + i] =
                (float)(rnd.NextDouble() * 2 * limit - limit); // Random value in [-limit, limit]
            var weightsOffset = i * NumInputs;

            for (var j = 0; j < NumInputs; j++)
                parameters[weightsOffset + j] =
                    (float)(rnd.NextDouble() * 2 * limit - limit); // Random value in [-limit, limit]
        }

        accelerator.Buffers.Parameters.View.SubView(LayerData.ParameterOffset, ParameterCount).CopyFromCPU(parameters);
    }

    public void Forward(NetworkAccelerator accelerator)
    {
        ForwardKernel(LayerData.NumOutputs * accelerator.Network.NetworkData.BatchSize, accelerator.Network.NetworkData,
            LayerData, accelerator.Buffers.Parameters.View, accelerator.Buffers.Activations.View);
    }

    public void Backward(NetworkAccelerator accelerator)
    {
        BackwardKernel1(LayerData.NumOutputs * LayerData.NumInputs * accelerator.Network.NetworkData.BatchSize,
            accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Parameters.View,
            accelerator.Buffers.Activations.View, accelerator.Buffers.Gradients.View,
            accelerator.Buffers.Errors.View);
        accelerator.accelerator.Synchronize();
        BackwardKernel2(LayerData.NumOutputs * accelerator.Network.NetworkData.BatchSize,
            accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Activations.View,
            accelerator.Buffers.Gradients.View,
            accelerator.Buffers.Errors.View);
    }

    public void AccumulateGradients(NetworkAccelerator accelerator)
    {
        GradientAccumulationKernel1(LayerData.NumOutputs * LayerData.NumInputs, accelerator.Network.NetworkData,
            LayerData, accelerator.Buffers.Parameters.View, accelerator.Buffers.Gradients.View,
            accelerator.Buffers.FirstMoment.View, accelerator.Buffers.SecondMoment.View);
        accelerator.accelerator.Synchronize();
        GradientAccumulationKernel2(LayerData.NumOutputs, accelerator.Network.NetworkData, LayerData,
            accelerator.Buffers.Parameters.View, accelerator.Buffers.Gradients.View,
            accelerator.Buffers.FirstMoment.View, accelerator.Buffers.SecondMoment.View);
    }

    public void CompileKernels(Accelerator accelerator)
    {
        ForwardKernel =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>>(
                    ForwardKernelImpl);
        BackwardKernel1 =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(BackwardKernel1Impl);

        BackwardKernel2 =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(BackwardKernel2Impl);

        GradientAccumulationKernel1 =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(GradientAccumulationKernel1Impl);

        GradientAccumulationKernel2 =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(GradientAccumulationKernel2Impl);
    }

    public LayerData LayerData { get; set; }

    // Forward pass kernel for calculating activations in a sigmoid layer
    public static void ForwardKernelImpl(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> parameters,
        ArrayView<float> activations)
    {
        // Compute indices for the batch and neuron within the batch
        var outputActivationIndex = index % layerData.NumOutputs;
        var batchIndex = index / layerData.NumOutputs;

        // Calculate offset for input and output activations in the batch
        var activationInputOffset = batchIndex * networkData.ActivationCount + layerData.ActivationInputOffset;
        var activationOutputOffset = batchIndex * networkData.ActivationCount + layerData.ActivationOutputOffset;

        // Start with the bias value as the initial sum for this neuron
        var sum = parameters[layerData.ParameterOffset + layerData.BiasOffset + outputActivationIndex];
        var weightsOffset = layerData.ParameterOffset + outputActivationIndex * layerData.NumInputs;

        // Accumulate the weighted sum for each input
        for (var j = 0; j < layerData.NumInputs; j++)
            sum += activations[activationInputOffset + j] * parameters[weightsOffset + j];

        // Apply sigmoid activation and store the result
        activations[activationOutputOffset + outputActivationIndex] = 1.0f / (1.0f + XMath.Exp(-sum));
    }

    // Backward pass kernel 1 for calculating weight gradients and propagating errors
    public static void BackwardKernel1Impl(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> parameters,
        ArrayView<float> activations,
        ArrayView<float> gradients,
        ArrayView<float> errors)
    {
        // Calculate indices for batch, output, and input
        int batchIndex = index / (layerData.NumInputs * layerData.NumOutputs);
        var outputIndex = index / layerData.NumInputs % layerData.NumOutputs;
        var inputIndex = index % layerData.NumInputs;

        // Calculate relevant offsets
        var activationInputOffset = batchIndex * networkData.ActivationCount + layerData.ActivationInputOffset;
        var activationOutputOffset = batchIndex * networkData.ActivationCount + layerData.ActivationOutputOffset;
        var currentErrorOffset = batchIndex * networkData.ErrorCount + layerData.CurrentLayerErrorOffset;
        var nextErrorOffset = batchIndex * networkData.ErrorCount + layerData.NextLayerErrorOffset;
        var gradientOffset = batchIndex * networkData.GradientCount + layerData.GradientOffset;

        // Calculate the derivative of the sigmoid activation
        var x = activations[activationOutputOffset + outputIndex];
        var derivative = x * (1.0f - x);
        var delta = errors[nextErrorOffset + outputIndex] * derivative;

        // Compute the offset for weights of this output neuron
        var weightsOffset = layerData.ParameterOffset + outputIndex * layerData.NumInputs;

        // Atomic update to propagate error back to current layer and accumulate weight gradient
        Atomic.Add(ref errors[currentErrorOffset + inputIndex], delta * parameters[weightsOffset + inputIndex]);
        gradients[gradientOffset + outputIndex * layerData.NumInputs + inputIndex] =
            delta * activations[activationInputOffset + inputIndex];
    }


    // Backward pass kernel 2 for updating bias gradients
    public static void BackwardKernel2Impl(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> activations,
        ArrayView<float> gradients,
        ArrayView<float> errors)
    {
        // Calculate indices and offsets
        var outputIndex = index % layerData.NumOutputs;
        var batchIndex = index / layerData.NumOutputs;
        var activationOutputOffset = batchIndex * networkData.ActivationCount + layerData.ActivationOutputOffset;
        var nextErrorOffset = batchIndex * networkData.ErrorCount + layerData.NextLayerErrorOffset;
        var gradientOffset = batchIndex * networkData.GradientCount + layerData.GradientOffset;

        // Compute delta using sigmoid derivative
        var x = activations[activationOutputOffset + outputIndex];
        var derivative = x * (1.0f - x);
        var delta = errors[nextErrorOffset + outputIndex] * derivative;

        // Update bias gradient for this neuron
        gradients[gradientOffset + layerData.BiasOffset + outputIndex] = delta;
    }

    // Kernel to accumulate and update weight gradients using Adam optimization
    public static void GradientAccumulationKernel1Impl(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> parameters,
        ArrayView<float> gradients,
        ArrayView<float> firstMoment,
        ArrayView<float> secondMoment)
    {
        // Calculate output and input indices
        var batchSize = networkData.BatchSize;
        var outputIndex = index / layerData.NumInputs;
        var inputIndex = index % layerData.NumInputs;

        // Initialize weight gradient accumulation
        var gradientIndex = layerData.GradientOffset + outputIndex * layerData.NumInputs + inputIndex;
        var weightGradient = 0.0f;

        // Accumulate weight gradients across the batch
        for (var i = 0; i < batchSize; i++) weightGradient += gradients[networkData.GradientCount * i + gradientIndex];

        weightGradient /= batchSize;

        // Update moments and bias-correct them
        firstMoment[gradientIndex] =
            networkData.Beta1 * firstMoment[gradientIndex] + (1 - networkData.Beta1) * weightGradient;
        secondMoment[gradientIndex] = networkData.Beta2 * secondMoment[gradientIndex] +
                                      (1 - networkData.Beta2) * weightGradient * weightGradient;
        var mHat = firstMoment[gradientIndex] / (1 - MathF.Pow(networkData.Beta1, networkData.Timestep));
        var vHat = secondMoment[gradientIndex] / (1 - MathF.Pow(networkData.Beta2, networkData.Timestep));

        // Update weight parameter
        var weightOffset = layerData.ParameterOffset + layerData.NumInputs * outputIndex + inputIndex;
        parameters[weightOffset] -= networkData.LearningRate * mHat / (MathF.Sqrt(vHat) + networkData.Epsilon);
    }

    // Kernel for accumulating and updating bias gradients using Adam optimization
    public static void GradientAccumulationKernel2Impl(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> parameters,
        ArrayView<float> gradients,
        ArrayView<float> firstMoment,
        ArrayView<float> secondMoment)
    {
        // Calculate batch size and bias gradient index
        var batchSize = networkData.BatchSize;
        var gradientIndex = layerData.GradientOffset + layerData.BiasOffset + index;

        // Accumulate bias gradients across the batch
        var biasGradient = 0.0f;
        for (var i = 0; i < batchSize; i++) biasGradient += gradients[networkData.GradientCount * i + gradientIndex];
        biasGradient /= batchSize;

        // Update first and second moments for bias
        firstMoment[gradientIndex] =
            networkData.Beta1 * firstMoment[gradientIndex] + (1 - networkData.Beta1) * biasGradient;
        secondMoment[gradientIndex] = networkData.Beta2 * secondMoment[gradientIndex] +
                                      (1 - networkData.Beta2) * biasGradient * biasGradient;

        // Bias-correct moments and apply update
        var mHat = firstMoment[gradientIndex] / (1 - MathF.Pow(networkData.Beta1, networkData.Timestep));
        var vHat = secondMoment[gradientIndex] / (1 - MathF.Pow(networkData.Beta2, networkData.Timestep));
        parameters[layerData.ParameterOffset + layerData.BiasOffset + index] -=
            networkData.LearningRate * mHat / (MathF.Sqrt(vHat) + networkData.Epsilon);
    }
}