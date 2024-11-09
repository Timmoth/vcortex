using vcortex.Layers;

namespace vcortex;

public class LayerData
{
    public readonly float[] Errors;
    public readonly float[] Output;

    public LayerData(ILayer layer)
    {
        Output = new float[layer.NumOutputs];
        Errors = new float[layer.NumInputs];
    }
}

public class NetworkData
{
    public readonly LayerData[] LayerData;

    public NetworkData(ILayer[] layers)
    {
        LayerData = new LayerData[layers.Length];
        for (var i = 0; i < layers.Length; i++) LayerData[i] = new LayerData(layers[i]);
    }
}

public class Network
{
    public readonly ILayer[] _layers;

    public Network(ILayer[] layers)
    {
        _layers = layers;
    }

    public float[] Predict(float[] inputs)
    {
        var networkData = new NetworkData(_layers);

        _layers[0].Forward(inputs, networkData.LayerData[0].Output);

        for (var index = 1; index < _layers.Length; index++)
            _layers[index].Forward(networkData.LayerData[index - 1].Output, networkData.LayerData[index].Output);

        return networkData.LayerData[^1].Output;
    }

    public float Train(List<(float[] inputs, float[] outputs)> trainData, NetworkData[] networkDataArray,
        float[][][] gradients, float learningRate = 0.02f)
    {
        var batchSize = trainData.Count;

        var sampleErrors = new float[batchSize]; // Store error for each sample

        // Perform forward and backward pass for each batch in parallel
        Parallel.For(0, batchSize, i =>
        {
            var networkData = networkDataArray[i];
            var inputs = trainData[i].inputs;
            var expectedOutputs = trainData[i].outputs;

            // Forward pass
            _layers[0].Forward(inputs, networkData.LayerData[0].Output);
            for (var j = 1; j < _layers.Length; j++)
                _layers[j].Forward(networkData.LayerData[j - 1].Output, networkData.LayerData[j].Output);

            // Calculate errors at the output layer for the current sample
            var finalLayer = networkData.LayerData[^1].Output;
            float sampleError = 0;
            var errors = new float[_layers[^1].NumOutputs];

            for (var j = 0; j < finalLayer.Length; j++)
            {
                errors[j] = (finalLayer[j] - expectedOutputs[j]);
                sampleError += errors[j] * errors[j]; // Sum of squared errors
            }

            sampleErrors[i] = sampleError / expectedOutputs.Length; // Store sample error

            // Backward pass for current sample
            var nextLayerErrors = errors;
            for (var j = _layers.Length - 1; j >= 0; j--)
            {
                var previousLayerOutput = j > 0 ? networkData.LayerData[j - 1].Output : inputs;
                _layers[j].Backward(previousLayerOutput, networkData.LayerData[j].Output,
                    networkData.LayerData[j].Errors, nextLayerErrors, gradients[j][i], learningRate);

                // Update errors for the next layer in the backward pass
                nextLayerErrors = networkData.LayerData[j].Errors;
            }
        });

        // Now accumulate gradients from all samples
        for (var layerIndex = 0; layerIndex < _layers.Length; layerIndex++)
            // Pass the gradients directly to AccumulateGradients
            _layers[layerIndex].AccumulateGradients(gradients[layerIndex], learningRate);

        // Compute average error for the batch
        var totalError = sampleErrors.Sum();
        return totalError;
    }

    public float Train(float[] inputs, float[] expectedOutputs, NetworkData networkData, float[][] gradients,
        float learningRate = 0.02f)
    {
        // Forward Pass
        _layers[0].Forward(inputs, networkData.LayerData[0].Output);
        for (var i = 1; i < _layers.Length; i++)
            _layers[i].Forward(networkData.LayerData[i - 1].Output, networkData.LayerData[i].Output);

        // Calculate Errors at Output Layer
        var finalLayer = networkData.LayerData[^1].Output;
        float sampleError = 0;
        var errors = new float[_layers[^1].NumOutputs];

        for (var i = 0; i < finalLayer.Length; i++)
        {
            errors[i] =finalLayer[i] - expectedOutputs[i];
            sampleError += errors[i] * errors[i];
        }


        var nextLayerErrors = errors;
        // Backward Pass
        for (var i = _layers.Length - 1; i >= 0; i--)
        {
            var previousLayerOutput = i > 0 ? networkData.LayerData[i - 1].Output : inputs;
            _layers[i].Backward(previousLayerOutput, networkData.LayerData[i].Output, networkData.LayerData[i].Errors,
                nextLayerErrors, gradients[i], learningRate);

            // Update errors for the next layer in the backward pass
            nextLayerErrors = networkData.LayerData[i].Errors;
        }

        for (var i = 0; i < _layers.Length; i++)
            _layers[i].AccumulateGradients(new float[1][]
            {
                gradients[i]
            }, learningRate);

        return sampleError / expectedOutputs.Length;
    }
}