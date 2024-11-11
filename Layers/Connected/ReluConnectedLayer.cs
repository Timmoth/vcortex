//using System.Numerics;
//using System.Runtime.CompilerServices;

//namespace vcortex.Layers.Connected;

//public class ReluConnectedLayer : IConnectedLayer
//{
//    public ReluConnectedLayer(int numOutputs)
//    {
//        NumOutputs = numOutputs;
//    }

//    public float[][] Weights { get; private set; }
//    public float[] Biases { get; private set; }

//    public int BiasOffset => NumInputs * NumOutputs;

//    public int NumInputs { get; private set; }
//    public int NumOutputs { get; }

//    public void Connect(ILayer prevLayer)
//    {
//        NumInputs = prevLayer.NumOutputs;
//        Biases = new float[NumOutputs];
//        Weights = new float[NumOutputs][];

//        for (var i = 0; i < NumOutputs; i++) Weights[i] = new float[NumInputs];
//    }

//    public void Connect(ConnectedInputConfig config)
//    {
//        NumInputs = config.NumInputs;
//        Biases = new float[NumOutputs];
//        Weights = new float[NumOutputs][];

//        for (var i = 0; i < NumOutputs; i++) Weights[i] = new float[NumInputs];
//    }

//    public int GradientCount => NumInputs * NumOutputs + NumOutputs;

//    public void Backward(float[] inputs, float[] outputs, float[] currentLayerErrors, float[] nextLayerErrors,
//        float[] gradients, float learningRate)
//    {
//        // Reset the current layer's errors
//        Array.Clear(currentLayerErrors, 0, currentLayerErrors.Length);

//        // Loop over each neuron in the output layer
//        for (var i = 0; i < NumOutputs; i++)
//        {
//            // Calculate delta for this neuron using the derivative of the activation function
//            var delta = nextLayerErrors[i] * Derivative(outputs[i]);
//            var weights = Weights[i];

//            // Inner loop to update weights and accumulate errors, element-by-element
//            for (var j = 0; j < NumInputs; j++)
//            {
//                // Update the error for the current input
//                currentLayerErrors[j] += delta * weights[j];

//                // Update the weight for this input
//                gradients[i * NumInputs + j] = delta * inputs[j];
//            }

//            // Update the bias for this neuron
//            gradients[BiasOffset + i] = delta;
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
//            var weights = Weights[i];

//            // Accumulate weight gradients
//            for (var j = 0; j < NumInputs; j++)
//            {
//                var gradientIndex = i * NumInputs + j;

//                // Accumulate the weight gradients across the batch
//                var weightGradient = 0.0f;
//                foreach (var gradient in gradients) weightGradient += gradient[gradientIndex];

//                // Average the gradient and apply the weight update
//                weights[j] -= lr * weightGradient;
//            }

//            // Accumulate and average the bias gradients
//            var biasGradient = 0.0f;
//            foreach (var gradient in
//                     gradients) biasGradient += gradient[BiasOffset + i]; // Assuming BiasOffset is correct

//            // Average the bias gradient and apply the update
//            Biases[i] -= lr * biasGradient;
//        }
//    }

//    public void FillRandom()
//    {
//        var rnd = Random.Shared;
//        var limit = MathF.Sqrt(6.0f / (NumInputs + NumOutputs));

//        for (var i = 0; i < NumOutputs; i++)
//        {
//            Biases[i] = (float)(rnd.NextDouble() * 2 * limit - limit); // Random value in [-limit, limit]
//            var weights = Weights[i];

//            for (var j = 0; j < NumInputs; j++)
//                weights[j] = (float)(rnd.NextDouble() * 2 * limit - limit); // Random value in [-limit, limit]
//        }
//    }

//    public void Forward(float[] activations, int offset)
//    {
//        var inputOffset = offset;
//        var outputOffset = offset + NumInputs;

//        for (var i = 0; i < NumOutputs; i++)
//        {
//            var sum = Biases[i];
//            var weights = Weights[i];

//            // Process each input element individually
//            for (var j = 0; j < NumInputs; j++)
//            {
//                sum += activations[offset + j] * weights[j];
//            }

//            // Apply the activation function
//            activations[outputOffset + i] = Activate(sum);
//        }
//    }

//    [MethodImpl(MethodImplOptions.AggressiveInlining)]
//    public float Activate(float x)
//    {
//        return Math.Max(0.0f, x);
//    }

//    [MethodImpl(MethodImplOptions.AggressiveInlining)]
//    public float Derivative(float x)
//    {
//        return x > 0 ? 1.0f : 0.0f;
//    }
//}