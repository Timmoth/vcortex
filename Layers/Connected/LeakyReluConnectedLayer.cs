//using System.Runtime.CompilerServices;

//namespace vcortex.Layers.Connected;

//public class LeakyReluConnectedLayer : IConnectedLayer
//{
//    private readonly float _alpha;


//    public LeakyReluConnectedLayer(int numOutputs, float alpha = 0.01f)
//    {
//        NumOutputs = numOutputs;
//        _alpha = alpha; // Small slope for negative inputs
//    }

//    public int BiasOffset => NumInputs * NumOutputs;

//    public int NumInputs { get; private set; }
//    public int NumOutputs { get; }
//    public int ActivationInputOffset { get; private set; }
//    public int ActivationOutputOffset { get; private set; }
//    public int CurrentLayerErrorOffset { get; private set; }
//    public int NextLayerErrorOffset { get; private set; }
//    public int GradientOffset { get; private set; }
//    public int ParameterCount { get; private set; }
//    public int ParameterOffset { get; private set; }
//    public float[] Parameters { get; set; }

//    public void Connect(ILayer prevLayer)
//    {
//        NumInputs = prevLayer.NumOutputs;
//        ParameterCount = NumOutputs * NumInputs + NumOutputs;
//        ParameterOffset = prevLayer.ParameterOffset + prevLayer.ParameterCount;

//        ActivationInputOffset = prevLayer.ActivationOutputOffset;
//        ActivationOutputOffset = prevLayer.ActivationOutputOffset + prevLayer.NumOutputs;
//        CurrentLayerErrorOffset = prevLayer.NextLayerErrorOffset;
//        NextLayerErrorOffset = prevLayer.CurrentLayerErrorOffset + NumInputs;
//        GradientOffset = prevLayer.GradientOffset + prevLayer.GradientCount;
//    }

//    public void Connect(ConnectedInputConfig config)
//    {
//        NumInputs = config.NumInputs;

//        ParameterCount = NumOutputs * NumInputs + NumOutputs;
//        ParameterOffset = 0;

//        ActivationInputOffset = 0;
//        ActivationOutputOffset = config.NumInputs;
//        CurrentLayerErrorOffset = 0;
//        NextLayerErrorOffset = config.NumInputs;
//        GradientOffset = 0;
//    }

//    public void FillRandom()
//    {
//        var rnd = Random.Shared;
//        var limit = MathF.Sqrt(6.0f / (NumInputs + NumOutputs));

//        for (var i = 0; i < NumOutputs; i++)
//        {
//            Parameters[ParameterOffset + BiasOffset + i] = (float)(rnd.NextDouble() * 2 * limit - limit); // Random value in [-limit, limit]
//            var weightsOffset = ParameterOffset + i * NumInputs;

//            for (var j = 0; j < NumInputs; j++)
//                Parameters[weightsOffset + j] = (float)(rnd.NextDouble() * 2 * limit - limit); // Random value in [-limit, limit]
//        }
//    }

//    public int GradientCount => NumInputs * NumOutputs + NumOutputs;

//    public void Backward(float[] activations, float[] errors,
//        float[] gradients, float learningRate)
//    {
//        // Reset the current layer's errors
//        Array.Clear(errors, CurrentLayerErrorOffset, NumInputs);

//        // Loop over each neuron in the output layer
//        for (var i = 0; i < NumOutputs; i++)
//        {
//            // Calculate delta for this neuron using the derivative of the activation function
//            var delta = errors[NextLayerErrorOffset + i] * Derivative(activations[ActivationOutputOffset + i]);
//            var weightsOffset = ParameterOffset + i * NumInputs;

//            // Inner loop to update weights and accumulate errors, element-by-element
//            for (var j = 0; j < NumInputs; j++)
//            {
//                // Update the error for the current input
//                errors[CurrentLayerErrorOffset + j] += delta * Parameters[weightsOffset + j];

//                // Update the weight for this input
//                gradients[GradientOffset + i * NumInputs + j] = delta * activations[ActivationInputOffset + j];
//            }

//            // Update the bias for this neuron
//            gradients[GradientOffset + BiasOffset + i] = delta;
//        }
//    }

//    public void AccumulateGradients(float[][] gradients, float learningRate)
//    {
//        // Number of samples in the batch
//        var lr = learningRate / gradients.Length;

//        // Loop over each output neuron
//        for (var i = 0; i < NumOutputs; i++)
//        {
//            // Update the weights for this neuron
//            var weightsOffset = ParameterOffset + i * NumInputs;

//            // Accumulate weight gradients
//            for (var j = 0; j < NumInputs; j++)
//            {
//                var gradientIndex = i * NumInputs + j;

//                // Accumulate the weight gradients across the batch
//                var weightGradient = 0.0f;
//                foreach (var gradient in gradients) weightGradient += gradient[GradientOffset + gradientIndex];

//                // Average the gradient and apply the weight update
//                Parameters[weightsOffset + j] -= lr * weightGradient;
//            }

//            // Accumulate and average the bias gradients
//            var biasGradient = 0.0f;
//            foreach (var gradient in gradients) biasGradient += gradient[GradientOffset + BiasOffset + i]; // Assuming BiasOffset is correct

//            // Average the bias gradient and apply the update
//            Parameters[ParameterOffset + BiasOffset + i] -= lr * biasGradient;
//        }
//    }

//    public void Forward(float[] activations)
//    {
//        for (var i = 0; i < NumOutputs; i++)
//        {
//            var sum = Parameters[ParameterOffset + BiasOffset + i];
//            var weightsOffset = ParameterOffset + i * NumInputs;

//            // Process each input element individually
//            for (var j = 0; j < NumInputs; j++)
//            {
//                sum += activations[ActivationInputOffset + j] * Parameters[weightsOffset + j];
//            }

//            // Apply the activation function
//            activations[ActivationOutputOffset + i] = Activate(sum);
//        }
//    }

//    [MethodImpl(MethodImplOptions.AggressiveInlining)]
//    public float Activate(float x)
//    {
//        // Leaky ReLU activation
//        return x > 0 ? x : _alpha * x;
//    }

//    [MethodImpl(MethodImplOptions.AggressiveInlining)]
//    public float Derivative(float x)
//    {
//        // Derivative of Leaky ReLU
//        return x > 0 ? 1.0f : _alpha;
//    }
//}

