using System.Numerics;
using System.Runtime.CompilerServices;

namespace vcortex.Layers.Connected;

public class LeakyReluConnectedLayer : IConnectedLayer
{
    private readonly float _alpha;


    public LeakyReluConnectedLayer(int numOutputs, float alpha = 0.01f)
    {
        NumOutputs = numOutputs;
        _alpha = alpha; // Small slope for negative inputs
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
        // Number of samples in the batch
        var lr = learningRate / gradients.Length;

        // Loop over each output neuron
        for (var i = 0; i < NumOutputs; i++)
        {
            // Update the weights for this neuron
            var weights = Weights[i];

            // Accumulate weight gradients
            for (var j = 0; j < NumInputs; j++)
            {
                var gradientIndex = i * NumInputs + j;

                // Accumulate the weight gradients across the batch
                var weightGradient = 0.0f;
                foreach (var gradient in gradients) weightGradient += gradient[gradientIndex];

                // Average the gradient and apply the weight update
                weights[j] -= lr * weightGradient;
            }

            // Accumulate and average the bias gradients
            var biasGradient = 0.0f;
            foreach (var gradient in gradients) biasGradient += gradient[BiasOffset + i]; // Assuming BiasOffset is correct

            // Average the bias gradient and apply the update
            Biases[i] -= lr * biasGradient;
        }
    }

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
        float learningRate)
    {
        Array.Clear(currentLayerErrors, 0, currentLayerErrors.Length);

        for (var i = 0; i < NumOutputs; i++)
        {
            var delta = nextLayerErrors[i] * Derivative(outputs[i]);
            var adjustedDelta = learningRate * delta;
            var weights = Weights[i];

            var simdLength = Vector<float>.Count;
            var j = 0;

            // Use SIMD for main part of the inner loop
            for (; j <= NumInputs - simdLength; j += simdLength)
            {
                var inputsVector = new Vector<float>(inputs, j);
                var weightsVector = new Vector<float>(weights, j);
                var deltaVector = new Vector<float>(delta);

                // Update weights in a vectorized way
                weightsVector += adjustedDelta * inputsVector;
                weightsVector.CopyTo(weights, j);

                // Accumulate errors in a vectorized way
                var errorContribution = deltaVector * weightsVector;
                var errorsVector = new Vector<float>(currentLayerErrors, j);
                errorsVector += errorContribution;
                errorsVector.CopyTo(currentLayerErrors, j);
            }

            // Process any remaining elements
            for (; j < NumInputs; j++)
            {
                currentLayerErrors[j] += delta * weights[j];
                weights[j] += adjustedDelta * inputs[j];
            }

            Biases[i] += adjustedDelta;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float Activate(float x)
    {
        // Leaky ReLU activation
        return x > 0 ? x : _alpha * x;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float Derivative(float x)
    {
        // Derivative of Leaky ReLU
        return x > 0 ? 1.0f : _alpha;
    }
}