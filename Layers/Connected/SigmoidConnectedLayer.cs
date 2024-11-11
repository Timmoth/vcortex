using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.IR;
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

    public int NumInputs { get; private set; }
    public int NumOutputs { get; }

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
            parameters[BiasOffset + i] =
                (float)(rnd.NextDouble() * 2 * limit - limit); // Random value in [-limit, limit]
            var weightsOffset = i * NumInputs;

            for (var j = 0; j < NumInputs; j++)
                parameters[weightsOffset + j] =
                    (float)(rnd.NextDouble() * 2 * limit - limit); // Random value in [-limit, limit]
        }

        accelerator.Buffers.Parameters.View.SubView(LayerData.ParameterOffset, ParameterCount).CopyFromCPU(parameters);
    }

    public int GradientCount => NumInputs * NumOutputs + NumOutputs;
    public int ActivationInputOffset { get; private set; }
    public int ActivationOutputOffset { get; private set; }
    public int CurrentLayerErrorOffset { get; private set; }
    public int NextLayerErrorOffset { get; private set; }
    public int GradientOffset { get; private set; }
    public int ParameterCount { get; private set; }
    public int ParameterOffset { get; private set; }
    public float[] Parameters { get; set; }

    public void Forward(float[] activations)
    {
        for (var i = 0; i < NumOutputs; i++)
        {
            var sum = Parameters[ParameterOffset + BiasOffset + i];
            var weightsOffset = ParameterOffset + i * NumInputs;

            // Process each input element individually
            for (var j = 0; j < NumInputs; j++)
                sum += activations[ActivationInputOffset + j] * Parameters[weightsOffset + j];

            // Apply the activation function to the sum and store it in the output
            activations[ActivationOutputOffset + i] = Activate(sum);
        }
    }
    public static void ForwardKernelImpl(
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

        var sum = parameters[layerData.ParameterOffset + layerData.BiasOffset + interBatchIndex];
        var weightsOffset = layerData.ParameterOffset + interBatchIndex * layerData.NumInputs;

        // Process each input element individually
        for (var j = 0; j < layerData.NumInputs; j++)
        {
            sum += activations[activationInputOffset + j] * parameters[weightsOffset + j];
        }

        // Apply the activation function to the sum and store it in the output
        activations[activationOutputOffset + interBatchIndex] = 1.0f / (1.0f + (float)Math.Exp(-sum));
    }

    public void Forward(NetworkAccelerator accelerator)
    {
        ForwardKernel(LayerData.NumOutputs * accelerator.Network.NetworkData.BatchSize, accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Parameters.View, accelerator.Buffers.Activations.View);
    }

    public void Backward(float[] activations, float[] errors,
        float[] gradients, float learningRate)
    {
        // Reset the current layer's errors
        Array.Clear(errors, CurrentLayerErrorOffset, NumInputs);

        // Loop over each neuron in the output layer
        for (var i = 0; i < NumOutputs; i++)
        {
            // Calculate delta for this neuron using the derivative of the activation function
            var delta = errors[NextLayerErrorOffset + i] * Derivative(activations[ActivationOutputOffset + i]);
            var weightsOffset = ParameterOffset + i * NumInputs;

            // Inner loop to update weights and accumulate errors, element-by-element
            for (var j = 0; j < NumInputs; j++)
            {
                // Update the error for the current input
                errors[CurrentLayerErrorOffset + j] += delta * Parameters[weightsOffset + j];

                // Update the weight for this input
                gradients[GradientOffset + i * NumInputs + j] = delta * activations[ActivationInputOffset + j];
            }

            // Update the bias for this neuron
            gradients[GradientOffset + BiasOffset + i] = delta;
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
        int batchIndex = index / (layerData.NumInputs * layerData.NumOutputs);         // First dimension (a)
        int outputIndex = (index / layerData.NumInputs) % layerData.NumOutputs;         // Second dimension (b)
        int inputIndex = index % layerData.NumInputs;
        var activationInputOffset = batchIndex * networkData.ActivationCount + layerData.ActivationInputOffset;
        var activationOutputOffset = batchIndex * networkData.ActivationCount + layerData.ActivationOutputOffset;
        var currentErrorOffset = batchIndex * networkData.ErrorCount + layerData.CurrentLayerErrorOffset;
        var nextErrorOffset = batchIndex * networkData.ErrorCount + layerData.NextLayerErrorOffset;
        var gradientOffset = batchIndex * networkData.GradientCount + layerData.GradientOffset;

        // Loop over each neuron in the output layer
        // Calculate delta for this neuron using the derivative of the activation function
        var x = activations[activationOutputOffset + outputIndex];
        var derivative = x * (1.0f - x);
        var delta = errors[nextErrorOffset + outputIndex] * derivative;
        var weightsOffset = layerData.ParameterOffset + outputIndex * layerData.NumInputs;

        // Inner loop to update weights and accumulate errors, element-by-element
        // Update the error for the current input
        Atomic.Add(ref errors[currentErrorOffset + inputIndex], delta * parameters[weightsOffset + inputIndex]);

        // Update the weight for this input
        gradients[gradientOffset + outputIndex * layerData.NumInputs + inputIndex] = delta * activations[activationInputOffset + inputIndex];
    }

    public static void BackwardKernel2Impl(
    Index1D index,
    NetworkData networkData,
    LayerData layerData,
    ArrayView<float> activations,
    ArrayView<float> gradients,
    ArrayView<float> errors)
    {
        var interBatchIndex = index % layerData.NumOutputs;
        var batchOffset = index / layerData.NumOutputs;
        var activationOutputOffset = batchOffset * networkData.ActivationCount + layerData.ActivationOutputOffset;
        var nextErrorOffset = batchOffset * networkData.ErrorCount + layerData.NextLayerErrorOffset;
        var gradientOffset = batchOffset * networkData.GradientCount + layerData.GradientOffset;

        // Loop over each neuron in the output layer
        // Calculate delta for this neuron using the derivative of the activation function
        var x = activations[activationOutputOffset + interBatchIndex];
        var derivative = x * (1.0f - x);
        var delta = errors[nextErrorOffset + interBatchIndex] * derivative;

        // Update the bias for this neuron
        gradients[gradientOffset + layerData.BiasOffset + interBatchIndex] = delta;
    }

    public void Backward(NetworkAccelerator accelerator)
    {
        BackwardKernel1(LayerData.NumOutputs * LayerData.NumInputs * accelerator.Network.NetworkData.BatchSize, accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Parameters.View, accelerator.Buffers.Activations.View, accelerator.Buffers.Gradients.View,
            accelerator.Buffers.Errors.View);
        BackwardKernel2(LayerData.NumOutputs * accelerator.Network.NetworkData.BatchSize, accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Activations.View, accelerator.Buffers.Gradients.View,
            accelerator.Buffers.Errors.View);
    }

    public void AccumulateGradients(float[][] gradients, float learningRate)
    {
        var batchSize = gradients.Length;
        var scaledLearningRate = learningRate / batchSize; // Scale learning rate by batch size for averaging

        // Loop over each output neuron
        for (var i = 0; i < NumOutputs; i++)
        {
            // Access weights for this neuron
            var weightsOffset = ParameterOffset + i * NumInputs;

            // Accumulate weight gradients
            for (var j = 0; j < NumInputs; j++)
            {
                var gradientIndex = i * NumInputs + j;

                // Sum gradients for this weight across the batch
                var totalWeightGradient = 0.0f;
                foreach (var gradient in gradients) totalWeightGradient += gradient[GradientOffset + gradientIndex];

                // Average the gradient and apply weight update
                Parameters[weightsOffset + j] -= scaledLearningRate * totalWeightGradient;
            }

            // Sum and average bias gradients
            var totalBiasGradient = 0.0f;
            foreach (var gradient in gradients) totalBiasGradient += gradient[GradientOffset + BiasOffset + i];

            // Apply averaged bias gradient update
            Parameters[ParameterOffset + BiasOffset + i] -= scaledLearningRate * totalBiasGradient;
        }
    }

    public static void GradientAccumulationKernel1Impl(
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
        var outputIndex = index / layerData.NumInputs;
        var inputIndex = index % layerData.NumInputs;

        // Loop over each output neuron
        // Update the weights for this neuron
        var weightOffset = layerData.ParameterOffset + layerData.NumInputs * outputIndex + inputIndex;

        // Accumulate weight gradients
        var gradientIndex = + layerData.GradientOffset + outputIndex * layerData.NumInputs + inputIndex;

        // Accumulate the weight gradients across the batch
        var weightGradient = 0.0f;
        for (int i = 0; i < batchSize; i++)
        {
            weightGradient += gradients[networkData.GradientCount * i + gradientIndex];
        }
        
        // Apply batch scaling factor
        weightGradient /= batchSize;

        // Update the first and second moment estimates
        firstMoment[gradientIndex] = layerData.Beta1 * firstMoment[gradientIndex] + (1 - layerData.Beta1) * weightGradient;
        secondMoment[gradientIndex] = layerData.Beta2 * secondMoment[gradientIndex] + (1 - layerData.Beta2) * weightGradient * weightGradient;

        // Bias correction for the moments
        var mHat = firstMoment[gradientIndex] / (1 - MathF.Pow(layerData.Beta1, layerData.Timestep));
        var vHat = secondMoment[gradientIndex] / (1 - MathF.Pow(layerData.Beta2, layerData.Timestep));
        
        // Average the gradient and apply the weight update
        parameters[weightOffset] -= networkData.LearningRate * mHat / (MathF.Sqrt(vHat) + layerData.Epsilon);
    }

    public static void GradientAccumulationKernel2Impl(
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
        var lr = networkData.LearningRate; // Scale learning rate by batch size for averaging
        var interBatchIndex = index;
        var gradientIndex = layerData.GradientOffset + layerData.BiasOffset + interBatchIndex;
        
        // Accumulate and average the bias gradients
        var biasGradient = 0.0f;
        for (int i = 0; i < batchSize; i++)
        {
            biasGradient += gradients[networkData.GradientCount * i + gradientIndex];
        }

        // Apply batch scaling factor
        biasGradient /= batchSize;

        // Update the first and second moment estimates
        firstMoment[gradientIndex] = layerData.Beta1 * firstMoment[gradientIndex] + (1 - layerData.Beta1) * biasGradient;
        secondMoment[gradientIndex] = layerData.Beta2 * secondMoment[gradientIndex] + (1 - layerData.Beta2) * biasGradient * biasGradient;

        // Bias correction for the moments
        var mHat = firstMoment[gradientIndex] / (1 - MathF.Pow(layerData.Beta1, layerData.Timestep));
        var vHat = secondMoment[gradientIndex] / (1 - MathF.Pow(layerData.Beta2, layerData.Timestep));
        
        // Average the gradient and apply the weight update
        parameters[layerData.ParameterOffset + layerData.BiasOffset + interBatchIndex] -= networkData.LearningRate * mHat / (MathF.Sqrt(vHat) + layerData.Epsilon);
        
    }

    public void AccumulateGradients(NetworkAccelerator accelerator)
    {
        GradientAccumulationKernel1(LayerData.NumOutputs * LayerData.NumInputs, accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Parameters.View, accelerator.Buffers.Gradients.View, accelerator.Buffers.FirstMoment.View, accelerator.Buffers.SecondMoment.View);
        GradientAccumulationKernel2(LayerData.NumOutputs, accelerator.Network.NetworkData, LayerData, accelerator.Buffers.Parameters.View, accelerator.Buffers.Gradients.View, accelerator.Buffers.FirstMoment.View, accelerator.Buffers.SecondMoment.View);
    }

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>> ForwardKernel { get; private set; }

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> BackwardKernel1 { get; private set; } 
    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> BackwardKernel2 { get; private set; }

    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>> GradientAccumulationKernel1 { get; private set; }
    public Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>> GradientAccumulationKernel2 { get; private set; }

    public void CompileKernels(Accelerator accelerator)
    {
        ForwardKernel =
            accelerator.LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>>(
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
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>>(GradientAccumulationKernel1Impl);

        GradientAccumulationKernel2 =
            accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>>(GradientAccumulationKernel2Impl);
    }

    public LayerData LayerData { get; set; }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float Activate(float x)
    {
        return 1.0f / (1.0f + (float)Math.Exp(-x));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float Derivative(float x)
    {
        return x * (1.0f - x);
    }
}