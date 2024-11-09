using System.Numerics;
using System.Runtime.CompilerServices;

namespace vcortex.Layers.Connected;

public class SigmoidConnectedLayer : IConnectedLayer
{
    public SigmoidConnectedLayer(int numOutputs)
    {
        NumOutputs = numOutputs;
    }

    public float[][] Weights { get; private set; }
    public float[] Biases { get; private set; }

    public int BiasOffset => NumInputs * NumOutputs;

    public int NumInputs { get; private set; }
    public int NumOutputs { get; }

    public void Connect(ILayer prevLayer)
    {
        NumInputs = prevLayer.NumOutputs;
        Biases = new float[NumOutputs];
        Weights = new float[NumOutputs][];

        for (var i = 0; i < NumOutputs; i++) Weights[i] = new float[NumInputs];
    }

    public void Connect(ConnectedInputConfig config)
    {
        NumInputs = config.NumInputs;
        Biases = new float[NumOutputs];
        Weights = new float[NumOutputs][];

        for (var i = 0; i < NumOutputs; i++) Weights[i] = new float[NumInputs];
    }

    public void FillRandom()
    {
        var rnd = Random.Shared;
        var limit = MathF.Sqrt(6.0f / (NumInputs + NumOutputs));

        for (var i = 0; i < NumOutputs; i++)
        {
            Biases[i] = (float)(rnd.NextDouble() * 2 * limit - limit); // Random value in [-limit, limit]
            var weights = Weights[i];

            for (var j = 0; j < NumInputs; j++)
                weights[j] = (float)(rnd.NextDouble() * 2 * limit - limit); // Random value in [-limit, limit]
        }
    }

    public int GradientCount => NumInputs * NumOutputs + NumOutputs;

    public void Forward(float[] inputs, float[] outputs)
    {
        for (var i = 0; i < NumOutputs; i++)
        {
            var sum = Biases[i];
            var weights = Weights[i];
            var simdLength = Vector<float>.Count;
            var j = 0;

            // Use SIMD for the main part of the loop
            for (; j <= NumInputs - simdLength; j += simdLength)
            {
                // Load chunks of inputs and weights into SIMD vectors
                var inputVector = new Vector<float>(inputs, j);
                var weightVector = new Vector<float>(weights, j);

                // Multiply and sum them into a single vector
                sum += Vector.Dot(inputVector, weightVector);
            }

            // Process remaining elements that didn’t fit into SIMD chunks
            for (; j < NumInputs; j++) sum += inputs[j] * weights[j];

            outputs[i] = Activate(sum);
        }
    }

    public void Backward(float[] inputs, float[] outputs, float[] currentLayerErrors, float[] nextLayerErrors,
        float[] gradients, float learningRate)
    {
        // Reset the current layer's errors
        Array.Clear(currentLayerErrors, 0, currentLayerErrors.Length);

        // Loop over each neuron in the output layer
        for (var i = 0; i < NumOutputs; i++)
        {
            // Calculate delta for this neuron using the derivative of the activation function
            var delta = nextLayerErrors[i] * Derivative(outputs[i]);
            var weights = Weights[i];

            // Inner loop to update weights and accumulate errors, element-by-element
            for (var j = 0; j < NumInputs; j++)
            {
                // Update the error for the current input
                currentLayerErrors[j] += delta * weights[j];

                // Update the weight for this input
                gradients[i * NumInputs + j] = delta * inputs[j];
            }

            // Update the bias for this neuron
            gradients[BiasOffset + i] = delta;
        }
    }

    public void AccumulateGradients(float[][] gradients, float learningRate)
    {
        var batchSize = gradients.Length;
        var scaledLearningRate = learningRate / batchSize; // Scale learning rate by batch size for averaging

        // Loop over each output neuron
        for (var i = 0; i < NumOutputs; i++)
        {
            // Access weights for this neuron
            var weights = Weights[i];

            // Accumulate weight gradients
            for (var j = 0; j < NumInputs; j++)
            {
                var gradientIndex = i * NumInputs + j;

                // Sum gradients for this weight across the batch
                var totalWeightGradient = 0.0f;
                foreach (var gradient in gradients) totalWeightGradient += gradient[gradientIndex];

                // Average the gradient and apply weight update
                weights[j] -= scaledLearningRate * totalWeightGradient;
            }

            // Sum and average bias gradients
            var totalBiasGradient = 0.0f;
            foreach (var gradient in gradients) totalBiasGradient += gradient[BiasOffset + i];

            // Apply averaged bias gradient update
            Biases[i] -= scaledLearningRate * totalBiasGradient;
        }
    }


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