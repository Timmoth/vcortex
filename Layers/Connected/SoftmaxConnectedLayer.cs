namespace vcortex.Layers.Connected;

public class SoftmaxConnectedLayer : IConnectedLayer
{
    public SoftmaxConnectedLayer(int numOutputs)
    {
        NumOutputs = numOutputs;
    }

    public float[][] Weights { get; private set; }
    public float[] Biases { get; private set; }
    public int BiasOffset => NumInputs * NumOutputs;

    public int NumInputs { get; private set; }
    public int NumOutputs { get; }
    public int GradientCount => NumInputs * NumOutputs + NumOutputs;

    public void Backward(float[] inputs, float[] outputs, float[] currentLayerErrors, float[] nextLayerErrors,
        float[] gradients,
        float learningRate)
    {
        Array.Clear(currentLayerErrors, 0, currentLayerErrors.Length);

        for (var i = 0; i < NumOutputs; i++)
        {
            var delta = nextLayerErrors[i]; // Assuming errors are gradients wrt softmax outputs
            var weights = Weights[i];

            for (var j = 0; j < NumInputs; j++)
            {
                currentLayerErrors[j] += delta * weights[j];
                gradients[i * NumInputs + j] = delta * inputs[j];
            }

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
            foreach (var gradient in
                     gradients) biasGradient += gradient[BiasOffset + i]; // Assuming BiasOffset is correct

            // Average the bias gradient and apply the update
            Biases[i] -= lr * biasGradient;
        }
    }

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


    public void Forward(float[] inputs, float[] outputs)
    {
        // Calculate the raw scores for each output
        for (var i = 0; i < NumOutputs; i++)
        {
            var sum = Biases[i];
            var weights = Weights[i];
            for (var j = 0; j < NumInputs; j++) sum += inputs[j] * weights[j];

            outputs[i] = sum;
        }

        // Apply stable softmax to the output scores
        ApplySoftmax(outputs);
    }

    private void ApplySoftmax(Span<float> outputs)
    {
        // Find the max value in outputs for numerical stability
        var maxVal = outputs[0];
        for (var i = 1; i < NumOutputs; i++)
            if (outputs[i] > maxVal)
                maxVal = outputs[i];

        // Calculate exponentials and the sum of exponentials
        float sumExp = 0;
        for (var i = 0; i < NumOutputs; i++)
        {
            // Apply stabilized exponentiation
            outputs[i] = (float)Math.Exp(outputs[i] - maxVal);

            // Check for infinity, indicating overflow
            if (float.IsInfinity(outputs[i]))
                throw new OverflowException($"Exp overflow detected in output at index {i}");

            sumExp += outputs[i];
        }

        // If sumExp is zero or not a valid number, return a uniform distribution
        if (sumExp <= 0 || float.IsNaN(sumExp) || float.IsInfinity(sumExp))
        {
            var uniformProb = 1.0f / NumOutputs;
            for (var i = 0; i < NumOutputs; i++) outputs[i] = uniformProb;
        }
        else
        {
            // Normalize to get probabilities
            for (var i = 0; i < NumOutputs; i++)
            {
                outputs[i] /= sumExp;

                // Check for NaN, which indicates a division by zero or an invalid operation
                if (float.IsNaN(outputs[i]))
                    throw new ArithmeticException($"NaN detected in softmax output at index {i}");
            }
        }
    }
}