using vcortex.Accelerated;
using vcortex.Layers;

namespace vcortex;

public class Network
{
    public readonly ILayer[] _layers;

    public int ActivationCount => _layers.Sum(l => l.NumOutputs) + _layers[0].NumInputs;
    public int GradientCount => _layers.Sum(l => l.GradientCount);
    public int ParameterCount => _layers.Sum(l => l.ParameterCount);

    public NetworkData NetworkData;
    public Network(ILayer[] layers)
    {
        _layers = layers;
        NetworkData = new NetworkData()
        {
            LearningRate = 0.01f,
            BatchSize = 200,
            ActivationCount = ActivationCount,
            ErrorCount = ActivationCount,
            GradientCount = GradientCount
        };
    }

    public float[] Predict(float[] inputs)
    {
        var activations = new float[ActivationCount];

        Array.Copy(inputs, activations, inputs.Length);

        foreach (var t in _layers)
            t.Forward(activations);

        var lastLayer = _layers[^1];
        return activations.AsSpan(lastLayer.ActivationOutputOffset, lastLayer.NumOutputs).ToArray();
    }

    public float Train(List<(float[] inputs, float[] outputs)> trainData, float[][] activations, float[][] errors,
        float[][] gradients, float learningRate = 0.02f)
    {
        var batchSize = trainData.Count;

        var sampleErrors = new float[batchSize]; // Store error for each sample

        // Perform forward and backward pass for each batch in parallel
        Parallel.For(0, batchSize, i =>
        {
            var networkData = activations[i];
            var inputs = trainData[i].inputs;
            var err = errors[i];
            var expectedOutputs = trainData[i].outputs;
            var g = gradients[i];
            // Forward pass
            Array.Copy(inputs, networkData, inputs.Length);

            foreach (var t in _layers)
                t.Forward(networkData);

            // Calculate errors at the output layer for the current sample
            var lastLayer = _layers[^1];
            var finalLayer = networkData.AsSpan(lastLayer.ActivationOutputOffset, lastLayer.NumOutputs).ToArray();
            float sampleError = 0;

            for (var j = 0; j < finalLayer.Length; j++)
            {
                var e = err[lastLayer.NextLayerErrorOffset + j] = (finalLayer[j] - expectedOutputs[j]);
                sampleError += e * e; // Sum of squared errors
            }

            sampleErrors[i] = sampleError / expectedOutputs.Length; // Store sample error

            // Backward pass for current sample
            for (var j = _layers.Length - 1; j >= 0; j--)
            {
                _layers[j].Backward(networkData,
                    err, g, learningRate);
            }
        });

        // Now accumulate gradients from all samples
        for (var layerIndex = 0; layerIndex < _layers.Length; layerIndex++)
            // Pass the gradients directly to AccumulateGradients
            _layers[layerIndex].AccumulateGradients(gradients, learningRate);

        // Compute average error for the batch
        var totalError = sampleErrors.Sum();
        return totalError;
    }

    public float Train(float[] inputs, float[] expectedOutputs, float[] activations, float[] errors, float[] gradients,
        float learningRate = 0.02f)
    {
        // Forward Pass
        Array.Copy(inputs, activations, inputs.Length);

        foreach (var t in _layers)
            t.Forward(activations);

        // Calculate Errors at Output Layer
        var lastLayer = _layers[^1];
        var finalLayer = activations.AsSpan(lastLayer.ActivationOutputOffset, lastLayer.NumOutputs).ToArray();
        float sampleError = 0;

        for (var j = 0; j < finalLayer.Length; j++)
        {
            var e = errors[lastLayer.NextLayerErrorOffset + j] = (finalLayer[j] - expectedOutputs[j]);
            sampleError += e * e; // Sum of squared errors
        }

        // Backward Pass
        for (var i = _layers.Length - 1; i >= 0; i--)
        {
            _layers[i].Backward(activations, errors, gradients, learningRate);
        }

        for (var i = 0; i < _layers.Length; i++)
            _layers[i].AccumulateGradients(new float[1][]
            {
                gradients
            }, learningRate);

        return sampleError / expectedOutputs.Length;
    }
}