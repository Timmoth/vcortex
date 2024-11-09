namespace vcortex.Layers.Connected;

public class DropoutConnectedLayer : IConnectedLayer
{
    private readonly Random _random;
    private bool[] _mask;

    public DropoutConnectedLayer(float dropoutRate = 0.5f)
    {
        DropoutRate = dropoutRate;
        _random = Random.Shared;
    }

    private float DropoutRate { get; }

    public int NumInputs { get; private set; }
    public int NumOutputs { get; private set; }
    public int GradientCount => 0;

    public void Connect(ILayer prevLayer)
    {
        NumInputs = NumOutputs = prevLayer.NumOutputs;
        _mask = new bool[NumOutputs];
    }

    public void Connect(ConnectedInputConfig config)
    {
        NumInputs = NumOutputs = config.NumInputs;
        _mask = new bool[NumOutputs];
    }

    public void Forward(float[] inputs, float[] outputs)
    {
        // During training, randomly drop neurons according to DropoutRate
        for (var i = 0; i < NumOutputs; i++)
        {
            // Determine whether to keep this neuron
            _mask[i] = _random.NextDouble() >= DropoutRate;

            // Apply mask: if neuron is kept, use the input; otherwise, set output to zero
            outputs[i] = _mask[i] ? inputs[i] : 0.0f;
        }
    }

    public void Backward(float[] inputs, float[] outputs, float[] currentLayerErrors, float[] nextLayerErrors,
        float[] gradients,
        float learningRate)
    {
        // Backpropagation for dropout passes through only for active neurons
        Array.Clear(currentLayerErrors, 0, currentLayerErrors.Length);

        for (var i = 0; i < NumOutputs; i++)
            // Only propagate errors for active neurons (those that were not dropped)
            currentLayerErrors[i] = _mask[i] ? nextLayerErrors[i] : 0.0f;
    }

    public void AccumulateGradients(float[][] gradients, float learningRate)
    {
    }
}