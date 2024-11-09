namespace vcortex.Layers.Connected;

public class GumbelSoftmaxLayer : IConnectedLayer
{
    private readonly float _temperature;

    public GumbelSoftmaxLayer(int numOutputs, float temperature = 1.0f)
    {
        NumOutputs = numOutputs;
        _temperature = temperature;
    }

    public float[][] Weights { get; private set; }
    public float[] Biases { get; private set; }
    private int BiasOffset => NumInputs * NumOutputs;

    public int NumInputs { get; private set; }
    public int NumOutputs { get; }
    public int GradientCount => NumInputs * NumOutputs + NumOutputs;

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
            Biases[i] = (float)(rnd.NextDouble() * 2 * limit - limit);
            var weights = Weights[i];
            for (var j = 0; j < NumInputs; j++) weights[j] = (float)(rnd.NextDouble() * 2 * limit - limit);
        }
    }

    public void Forward(float[] inputs, float[] outputs)
    {
        // Calculate the logits (raw scores) for each output
        for (var i = 0; i < NumOutputs; i++)
        {
            var sum = Biases[i];
            var weights = Weights[i];
            for (var j = 0; j < NumInputs; j++) sum += inputs[j] * weights[j];
            outputs[i] = sum;
        }

        // Apply Gumbel-Softmax to the outputs
        ApplyGumbelSoftmax(outputs, _temperature);
    }

    public void Backward(float[] inputs, float[] outputs, float[] currentLayerErrors, float[] nextLayerErrors,
        float[] gradients, float learningRate)
    {
        Array.Clear(currentLayerErrors, 0, currentLayerErrors.Length);

        for (var i = 0; i < NumOutputs; i++)
        {
            var delta = nextLayerErrors[i];
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
        var lr = learningRate / gradients.Length;

        for (var i = 0; i < NumOutputs; i++)
        {
            var weights = Weights[i];

            for (var j = 0; j < NumInputs; j++)
            {
                float weightGradientSum = 0;
                foreach (var g in gradients) weightGradientSum += g[i * NumInputs + j];

                weights[j] -= lr * weightGradientSum;
            }

            float biasGradientSum = 0;
            foreach (var g in gradients) biasGradientSum += g[BiasOffset + i];

            Biases[i] -= lr * biasGradientSum;
        }
    }

    private void ApplyGumbelSoftmax(float[] logits, float temperature)
    {
        // Sample Gumbel noise and add it to the logits
        var gumbelNoise = SampleGumbel(logits.Length);
        for (var i = 0; i < logits.Length; i++) logits[i] = (logits[i] + gumbelNoise[i]) / temperature;

        // Apply softmax to the logits + noise
        Softmax(logits);
    }

    private void Softmax(float[] outputs)
    {
        var maxVal = outputs.Max();
        float sumExp = 0;

        for (var i = 0; i < outputs.Length; i++)
        {
            outputs[i] = (float)Math.Exp(outputs[i] - maxVal);
            sumExp += outputs[i];
        }

        for (var i = 0; i < outputs.Length; i++) outputs[i] /= sumExp;
    }

    private float[] SampleGumbel(int size)
    {
        var rnd = Random.Shared;
        var gumbelNoise = new float[size];

        for (var i = 0; i < size; i++)
        {
            var u = (float)rnd.NextDouble();
            gumbelNoise[i] = -MathF.Log(-MathF.Log(u + 1e-10f) + 1e-10f); // Small epsilon to prevent log(0)
        }

        return gumbelNoise;
    }
}