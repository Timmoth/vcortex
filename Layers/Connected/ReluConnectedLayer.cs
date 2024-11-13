using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using vcortex.Accelerated;

namespace vcortex.Layers.Connected;

public class ReluConnectedLayer : IConnectedLayer
{
    public ReluConnectedLayer(int numOutputs)
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

    public void CompileKernels(NetworkAccelerator accelerator)
    {
        ForwardKernel =
            accelerator.accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>>(
                    ForwardKernelImpl);
        BackwardKernel1 =
            accelerator.accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(BackwardKernel1Impl);

        BackwardKernel2 =
            accelerator.accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(BackwardKernel2Impl);

        GradientAccumulationKernel1 =
            accelerator.accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(GradientAccumulationKernel1Impl);

        GradientAccumulationKernel2 =
            accelerator.accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(GradientAccumulationKernel2Impl);
    }

    public LayerData LayerData { get; set; }

    // Forward pass kernel for calculating activations in a ReLU layer
    public static void ForwardKernelImpl(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> parameters,
        ArrayView<float> activations)
    {
        // Compute indices for the batch and neuron within the batch
        var batchIndex = index / layerData.NumOutputs;
        var outputIndex = index % layerData.NumOutputs;

        // Calculate offset for input and output activations in the batch
        var activationInputOffset = batchIndex * networkData.ActivationCount + layerData.ActivationInputOffset;
        var activationOutputOffset = batchIndex * networkData.ActivationCount + layerData.ActivationOutputOffset;

        // Initialize sum with bias for the output neuron
        var sum = parameters[layerData.ParameterOffset + layerData.BiasOffset + outputIndex];
        var weightsOffset = layerData.ParameterOffset + outputIndex * layerData.NumInputs;

        // Accumulate the weighted sum for each input connection
        for (var j = 0; j < layerData.NumInputs; j++)
        {
            sum += activations[activationInputOffset + j] * parameters[weightsOffset + j];
        }

        // Apply ReLU activation and store the result
        activations[activationOutputOffset + outputIndex] = XMath.Max(0.0f, sum);
    }


    // Backward pass kernel 1: Calculate weight gradients and propagate errors
    public static void BackwardKernel1Impl(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> parameters,
        ArrayView<float> activations,
        ArrayView<float> gradients,
        ArrayView<float> errors)
    {
        // Calculate indices for batch, output, and input neurons
        var batchIndex = index / (layerData.NumInputs * layerData.NumOutputs);
        var outputIndex = (index / layerData.NumInputs) % layerData.NumOutputs;
        var inputIndex = index % layerData.NumInputs;

        // Calculate offsets for activations, errors, and gradients
        var activationInputOffset = batchIndex * networkData.ActivationCount + layerData.ActivationInputOffset;
        var activationOutputOffset = batchIndex * networkData.ActivationCount + layerData.ActivationOutputOffset;
        var currentErrorOffset = batchIndex * networkData.ErrorCount + layerData.CurrentLayerErrorOffset;
        var nextErrorOffset = batchIndex * networkData.ErrorCount + layerData.NextLayerErrorOffset;
        var gradientOffset = layerData.GradientOffset;

        // Get output activation and apply ReLU derivative
        var outputActivation = activations[activationOutputOffset + outputIndex];
        var derivative = outputActivation > 0 ? 1.0f : 0.0f;
        var delta = errors[nextErrorOffset + outputIndex] * derivative;

        // Compute weight index for this output neuron
        var weightIndex = outputIndex * layerData.NumInputs + inputIndex;

        // Backpropagate error to current layer and accumulate weight gradient
        Atomic.Add(ref errors[currentErrorOffset + inputIndex], delta * parameters[layerData.ParameterOffset + weightIndex]);

        // Accumulate weight gradient
        gradients[gradientOffset + weightIndex] = delta * activations[activationInputOffset + inputIndex];
    }



    // Backward pass kernel 2: Calculate bias gradients
    public static void BackwardKernel2Impl(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> activations,
        ArrayView<float> gradients,
        ArrayView<float> errors)
    {
        // Calculate indices and offsets
        var batchIndex = index / layerData.NumOutputs;
        var outputIndex = index % layerData.NumOutputs;
        var activationOutputOffset = batchIndex * networkData.ActivationCount + layerData.ActivationOutputOffset;
        var nextErrorOffset = batchIndex * networkData.ErrorCount + layerData.NextLayerErrorOffset;
        var gradientOffset = layerData.GradientOffset;

        // Apply ReLU derivative to output activation
        var outputActivation = activations[activationOutputOffset + outputIndex];
        var derivative = outputActivation > 0 ? 1.0f : 0.0f;
        var delta = errors[nextErrorOffset + outputIndex] * derivative;

        // Update bias gradient for this output neuron
        gradients[gradientOffset + layerData.BiasOffset + outputIndex] = delta;
    }

    // Gradient accumulation kernel 1: Update weight gradients using Adam optimizer
    public static void GradientAccumulationKernel1Impl(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> parameters,
        ArrayView<float> gradients,
        ArrayView<float> firstMoment,
        ArrayView<float> secondMoment)
    {
        // Calculate output and input indices for this weight
        var outputIndex = index / layerData.NumInputs;
        var inputIndex = index % layerData.NumInputs;

        // Calculate gradient index and initialize accumulation for this weight
        var gradientIndex = layerData.GradientOffset + outputIndex * layerData.NumInputs + inputIndex;
        var weightGradient = 0.0f;

        // Accumulate gradient across the batch
        for (var i = 0; i < networkData.BatchSize; i++)
        {
            weightGradient += gradients[networkData.GradientCount * i + gradientIndex];
        }

        // Average gradient and apply Adam optimizer update
        weightGradient /= networkData.BatchSize;
        firstMoment[gradientIndex] = networkData.Beta1 * firstMoment[gradientIndex] + (1 - networkData.Beta1) * weightGradient;
        secondMoment[gradientIndex] = networkData.Beta2 * secondMoment[gradientIndex] + (1 - networkData.Beta2) * weightGradient * weightGradient;
        var mHat = firstMoment[gradientIndex] / (1 - MathF.Pow(networkData.Beta1, networkData.Timestep));
        var vHat = secondMoment[gradientIndex] / (1 - MathF.Pow(networkData.Beta2, networkData.Timestep));
        parameters[layerData.ParameterOffset + outputIndex * layerData.NumInputs + inputIndex] -= networkData.LearningRate * mHat / (MathF.Sqrt(vHat) + networkData.Epsilon);
    }


    // Gradient accumulation kernel 2: Update bias gradients using Adam optimizer
    public static void GradientAccumulationKernel2Impl(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> parameters,
        ArrayView<float> gradients,
        ArrayView<float> firstMoment,
        ArrayView<float> secondMoment)
    {
        // Calculate the gradient index for this bias
        var gradientIndex = layerData.GradientOffset + layerData.BiasOffset + index;

        // Accumulate gradient across the batch
        var biasGradient = 0.0f;
        for (var i = 0; i < networkData.BatchSize; i++)
        {
            biasGradient += gradients[networkData.GradientCount * i + gradientIndex];
        }

        // Average gradient and apply Adam optimizer update
        biasGradient /= networkData.BatchSize;
        firstMoment[gradientIndex] = networkData.Beta1 * firstMoment[gradientIndex] + (1 - networkData.Beta1) * biasGradient;
        secondMoment[gradientIndex] = networkData.Beta2 * secondMoment[gradientIndex] + (1 - networkData.Beta2) * biasGradient * biasGradient;
        var mHat = firstMoment[gradientIndex] / (1 - MathF.Pow(networkData.Beta1, networkData.Timestep));
        var vHat = secondMoment[gradientIndex] / (1 - MathF.Pow(networkData.Beta2, networkData.Timestep));
        parameters[layerData.ParameterOffset + layerData.BiasOffset + index] -= networkData.LearningRate * mHat / (MathF.Sqrt(vHat) + networkData.Epsilon);
    }

}