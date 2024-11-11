using ILGPU.Runtime;
using ILGPU;
using vcortex.Accelerated;
using ILGPU.Algorithms;

namespace vcortex.Layers.Connected;

public class SoftmaxConnectedLayer : IConnectedLayer
{
    public SoftmaxConnectedLayer(int numOutputs)
    {
        NumOutputs = numOutputs;
    }

    public int BiasOffset => NumInputs * NumOutputs;

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
    public float[] Parameters { get; set; }

    public void Backward(float[] activations, float[] errors,
        float[] gradients,
        float learningRate)
    {
        Array.Clear(errors, CurrentLayerErrorOffset, NumInputs);

        for (var i = 0; i < NumOutputs; i++)
        {
            var delta = errors[NextLayerErrorOffset + i]; // Assuming errors are gradients wrt softmax outputs
            var weightOffset = ParameterOffset + NumInputs * i;

            for (var j = 0; j < NumInputs; j++)
            {
                errors[CurrentLayerErrorOffset + j] += delta * Parameters[weightOffset + j];
                gradients[GradientOffset + i * NumInputs + j] = delta * activations[ActivationInputOffset + j];
            }

            gradients[GradientOffset + BiasOffset + i] = delta;
        }
    }

    public void AccumulateGradients(float[][] gradients, float learningRate)
    {
        // Number of samples in the batch
        var lr = learningRate / gradients.Length;

        // Loop over each output neuron
        for (var i = 0; i < NumOutputs; i++)
        {
            // Update the weights for this neuron
            var weightOffset = ParameterOffset + NumInputs * i;

            // Accumulate weight gradients
            for (var j = 0; j < NumInputs; j++)
            {
                var gradientIndex = i * NumInputs + j;

                // Accumulate the weight gradients across the batch
                var weightGradient = 0.0f;
                foreach (var gradient in gradients) weightGradient += gradient[GradientOffset + gradientIndex];

                // Average the gradient and apply the weight update
                Parameters[weightOffset + j] -= lr * weightGradient;
            }

            // Accumulate and average the bias gradients
            var biasGradient = 0.0f;
            foreach (var gradient in
                     gradients) biasGradient += gradient[GradientOffset + BiasOffset + i]; // Assuming BiasOffset is correct

            // Average the bias gradient and apply the update
            Parameters[ParameterOffset + BiasOffset + i] -= lr * biasGradient;
        }
    }

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

        LayerData = new LayerData()
        {
            ActivationInputOffset = ActivationInputOffset,
            ActivationOutputOffset = ActivationOutputOffset,
            CurrentLayerErrorOffset = CurrentLayerErrorOffset,
            NextLayerErrorOffset = NextLayerErrorOffset,
            BiasOffset = BiasOffset,
            GradientOffset = GradientOffset,
            NumInputs = NumInputs,
            NumOutputs = NumOutputs
        };
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

        LayerData = new LayerData()
        {
            ActivationInputOffset = ActivationInputOffset,
            ActivationOutputOffset = ActivationOutputOffset,
            CurrentLayerErrorOffset = CurrentLayerErrorOffset,
            NextLayerErrorOffset = NextLayerErrorOffset,
            BiasOffset = BiasOffset,
            GradientOffset = GradientOffset,
            NumInputs = NumInputs,
            NumOutputs = NumOutputs
        };
    }

    public virtual void FillRandom(NetworkAccelerator accelerator)
    {
        var rnd = Random.Shared;
        var limit = MathF.Sqrt(6.0f / (NumInputs + NumOutputs));

        var parameters = new float[ParameterCount];

        for (var i = 0; i < NumOutputs; i++)
        {
            parameters[BiasOffset + i] = (float)(rnd.NextDouble() * 2 * limit - limit); // Random value in [-limit, limit]
            var weightsOffset = NumInputs * i;

            for (var j = 0; j < NumInputs; j++)
                parameters[weightsOffset + j] = (float)(rnd.NextDouble() * 2 * limit - limit); // Random value in [-limit, limit]
        }

        accelerator.Buffers.Parameters.View.SubView(LayerData.ParameterOffset, ParameterCount).CopyFromCPU(parameters);
    }

    public void Forward(float[] activations)
    {
        // Calculate the raw scores for each output
        for (var i = 0; i < NumOutputs; i++)
        {
            var sum = Parameters[ParameterOffset + BiasOffset + i];
            var weightsOffset = ParameterOffset + NumInputs * i;
            for (var j = 0; j < NumInputs; j++) sum += activations[ActivationInputOffset + j] * Parameters[weightsOffset + j];

            activations[ActivationOutputOffset + i] = sum;
        }

        // Apply stable softmax to the output scores
        // Find the max value in outputs for numerical stability
        var maxVal = activations[ActivationOutputOffset];
        for (var i = 1; i < NumOutputs; i++)
            if (activations[ActivationOutputOffset + i] > maxVal)
                maxVal = activations[ActivationOutputOffset + i];

        // Calculate exponentials and the sum of exponentials
        float sumExp = 0;
        for (var i = 0; i < NumOutputs; i++)
        {
            // Apply stabilized exponentiation
            activations[ActivationOutputOffset + i] = XMath.Exp(activations[ActivationOutputOffset + i] - maxVal);

            // Check for infinity, indicating overflow
            if (float.IsInfinity(activations[ActivationOutputOffset + i]))
                throw new OverflowException($"Exp overflow detected in output at index {i}");

            sumExp += activations[ActivationOutputOffset + i];
        }

        // If sumExp is zero or not a valid number, return a uniform distribution
        if (sumExp <= 0 || float.IsNaN(sumExp) || float.IsInfinity(sumExp))
        {
            var uniformProb = 1.0f / NumOutputs;
            for (var i = 0; i < NumOutputs; i++) activations[ActivationOutputOffset + i] = uniformProb;
        }
        else
        {
            // Normalize to get probabilities
            for (var i = 0; i < NumOutputs; i++)
            {
                activations[ActivationOutputOffset + i] /= sumExp;

                // Check for NaN, which indicates a division by zero or an invalid operation
                if (float.IsNaN(activations[ActivationOutputOffset + i]))
                    throw new ArithmeticException($"NaN detected in softmax output at index {i}");
            }
        }

    }

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>> ForwardKernel1 { get; private set; }
    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>> ForwardKernel2 { get; private set; }

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> BackwardKernel1
    { get; private set; }
    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> BackwardKernel2
    { get; private set; }
    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>> GradientAccumulationKernel
    { get; private set; }


    public void CompileKernels(Accelerator accelerator)
    {
        ForwardKernel1 =
            accelerator.LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>>(
                ForwardKernel1Impl);

        ForwardKernel2 =
            accelerator.LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>>(
                ForwardKernel2Impl);

        BackwardKernel1 =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(BackwardKernel1Impl);

        BackwardKernel2 =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, ArrayView<float>>(BackwardKernel2Impl);

        GradientAccumulationKernel =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>>(GradientAccumulationKernelImpl);
    }

    public void Forward(NetworkAccelerator accelerator)
    {
        ForwardKernel1(LayerData.NumOutputs * accelerator.Network.NetworkData.BatchSize, accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Parameters.View, accelerator.Buffers.Activations.View);
        ForwardKernel2(accelerator.Network.NetworkData.BatchSize, accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Parameters.View, accelerator.Buffers.Activations.View);
    }

    public static void ForwardKernel1Impl(
Index1D index,
NetworkData networkData,
LayerData layerData,
ArrayView<float> parameters,
ArrayView<float> activations)
    {
        var interBatchIndex = index % layerData.NumOutputs;
        var batchOffset = index / layerData.NumOutputs;
        var activationInputOffset = batchOffset * networkData.ActivationCount + layerData.ActivationInputOffset;
        var activationOutputOffset = batchOffset * networkData.ActivationCount + layerData.ActivationOutputOffset;

        // Calculate the raw scores for each output
        var sum = parameters[layerData.ParameterOffset + layerData.BiasOffset + interBatchIndex];
        var weightsOffset = layerData.ParameterOffset + layerData.NumInputs * interBatchIndex;
        for (var j = 0; j < layerData.NumInputs; j++) sum += activations[activationInputOffset + j] * parameters[weightsOffset + j];

        activations[activationOutputOffset + interBatchIndex] = sum;
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

        // Apply stable softmax to the output scores
        // Find the max value in outputs for numerical stability
        var maxVal = activations[activationOutputOffset];
        for (var i = 1; i < layerData.NumOutputs; i++)
            if (activations[activationOutputOffset + i] > maxVal)
                maxVal = activations[activationOutputOffset + i];

        // Calculate exponentials and the sum of exponentials
        float sumExp = 0;
        for (var i = 0; i < layerData.NumOutputs; i++)
        {
            // Apply stabilized exponentiation
            activations[activationOutputOffset + i] = (float)XMath.Exp(activations[activationOutputOffset + i] - maxVal);

            sumExp += activations[activationOutputOffset + i];
        }

        // If sumExp is zero or not a valid number, return a uniform distribution
        if (sumExp <= 0 || float.IsNaN(sumExp) || float.IsInfinity(sumExp))
        {
            var uniformProb = 1.0f / layerData.NumOutputs;
            for (var i = 0; i < layerData.NumOutputs; i++) activations[activationOutputOffset + i] = uniformProb;
        }
        else
        {
            // Normalize to get probabilities
            for (var i = 0; i < layerData.NumOutputs; i++)
            {
                activations[activationOutputOffset + i] /= sumExp;
            }
        }
    }

    public void Backward(NetworkAccelerator accelerator)
    {
        BackwardKernel1(LayerData.NumOutputs * LayerData.NumInputs * accelerator.Network.NetworkData.BatchSize, accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Parameters.View, accelerator.Buffers.Activations.View, accelerator.Buffers.Gradients.View,
            accelerator.Buffers.Errors.View);
        BackwardKernel2(LayerData.NumOutputs * accelerator.Network.NetworkData.BatchSize, accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Parameters.View, accelerator.Buffers.Activations.View, accelerator.Buffers.Gradients.View,
            accelerator.Buffers.Errors.View);
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
        int batchIndex = index / (layerData.NumInputs * layerData.NumOutputs);         // First dimension (a)
        int outputIndex = (index / layerData.NumInputs) % layerData.NumOutputs;         // Second dimension (b)
        int inputIndex = index % layerData.NumInputs;
        var activationInputOffset = batchIndex * networkData.ActivationCount + layerData.ActivationInputOffset;
        var currentErrorOffset = batchIndex * networkData.ErrorCount + layerData.CurrentLayerErrorOffset;
        var nextErrorOffset = batchIndex * networkData.ErrorCount + layerData.NextLayerErrorOffset;
        var gradientOffset = batchIndex * networkData.GradientCount + layerData.GradientOffset;

        // Loop over each neuron in the output layer
        var delta = errors[nextErrorOffset + outputIndex]; // Assuming errors are gradients wrt softmax outputs
        var weightOffset = layerData.ParameterOffset + layerData.NumInputs * outputIndex;

        Atomic.Add(ref errors[currentErrorOffset + inputIndex], delta * parameters[weightOffset + inputIndex]);
        
        gradients[gradientOffset + outputIndex * layerData.NumInputs + inputIndex] = delta * activations[activationInputOffset + inputIndex];
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
        var interBatchIndex = index % layerData.NumOutputs;
        var batchOffset = index / layerData.NumOutputs;
        var nextErrorOffset = batchOffset * networkData.ErrorCount + layerData.NextLayerErrorOffset;
        var gradientOffset = batchOffset * networkData.GradientCount + layerData.GradientOffset;

        // Loop over each neuron in the output layer
        var delta = errors[nextErrorOffset + interBatchIndex]; // Assuming errors are gradients wrt softmax outputs
        var weightOffset = layerData.ParameterOffset + layerData.NumInputs * interBatchIndex;

        gradients[gradientOffset + layerData.BiasOffset + interBatchIndex] = delta;
    }

    public void AccumulateGradients(NetworkAccelerator accelerator)
    {
        GradientAccumulationKernel(LayerData.NumOutputs, accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Parameters.View, accelerator.Buffers.Gradients.View, accelerator.Buffers.FirstMoment.View, accelerator.Buffers.SecondMoment.View);
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
        // Number of samples in the batch
        var batchSize = networkData.BatchSize;
        var interBatchIndex = index;

        // Loop over each output neuron
        // Update the weights for this neuron
        var weightOffset = layerData.ParameterOffset + layerData.NumInputs * interBatchIndex;

        // Accumulate weight gradients
        for (var j = 0; j < layerData.NumInputs; j++)
        {
            var gradientIndex = layerData.GradientOffset + interBatchIndex * layerData.NumInputs + j;

            // Accumulate the weight gradients across the batch
            var weightGradient = 0.0f;
            for (int i = 0; i < batchSize; i++)
            {
                weightGradient += gradients[networkData.GradientCount * i  + gradientIndex];
            }
            
            weightGradient /= networkData.BatchSize; // Take the mean over the batch
            
            // Update the first and second moment estimates
            firstMoment[gradientIndex] = layerData.Beta1 * firstMoment[gradientIndex] + (1 - layerData.Beta1) * weightGradient;
            secondMoment[gradientIndex] = layerData.Beta2 * secondMoment[gradientIndex] + (1 - layerData.Beta2) * weightGradient * weightGradient;

            // Bias correction for the moments
            var mHat = firstMoment[gradientIndex] / (1 - MathF.Pow(layerData.Beta1, layerData.Timestep));
            var vHat = secondMoment[gradientIndex] / (1 - MathF.Pow(layerData.Beta2, layerData.Timestep));
        
            // Average the gradient and apply the weight update
            parameters[weightOffset + j] -= networkData.LearningRate * mHat / (MathF.Sqrt(vHat) + layerData.Epsilon);
        }

        var gradientIndex2 = layerData.GradientOffset + layerData.BiasOffset + interBatchIndex;
        // Accumulate and average the bias gradients
        var biasGradient = 0.0f;
        for (int i = 0; i < batchSize; i++)
        {
            biasGradient += gradients[networkData.GradientCount * i + gradientIndex2];
        }

        biasGradient /= networkData.BatchSize; // Take the mean over the batch
        
        // Update the first and second moment estimates
        firstMoment[gradientIndex2] = layerData.Beta1 * firstMoment[gradientIndex2] + (1 - layerData.Beta1) * biasGradient;
        secondMoment[gradientIndex2] = layerData.Beta2 * secondMoment[gradientIndex2] + (1 - layerData.Beta2) * biasGradient * biasGradient;

        // Bias correction for the moments
        var mHat2 = firstMoment[gradientIndex2] / (1 - MathF.Pow(layerData.Beta1, layerData.Timestep));
        var vHat2 = secondMoment[gradientIndex2] / (1 - MathF.Pow(layerData.Beta2, layerData.Timestep));
        
        // Average the gradient and apply the weight update
        parameters[layerData.ParameterOffset + layerData.BiasOffset + interBatchIndex] -= networkData.LearningRate * mHat2 / (MathF.Sqrt(vHat2) + layerData.Epsilon);
    }

    public LayerData LayerData { get; set; }

}