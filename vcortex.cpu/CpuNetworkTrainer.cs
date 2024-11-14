using System.Diagnostics;
using vcortex.cpu.Layers;
using vcortex.cpu.LossFunctions;
using vcortex.cpu.Optimizers;
using vcortex.Layers;
using vcortex.Network;
using vcortex.Optimizers;
using vcortex.Training;

namespace vcortex.cpu;

public class CpuNetworkTrainer : ICpuNetworkAgent
{
    private readonly ILayer[] _layers;
    private readonly IOptimizer _optimizer;

    public CpuNetworkTrainer(NetworkConfig network, TrainConfig trainingConfig)
    {
        Network = network;
        _trainConfig = trainingConfig;
        Buffers = new NetworkAcceleratorBuffers(network, trainingConfig.BatchSize);
        _optimizer = CpuOptimizerFactory.Create(trainingConfig.Optimizer, Buffers, network.NetworkData);
        _layers = network.Layers.Select(l => CpuLayerFactory.Create(l, Buffers, network.NetworkData)).ToArray();

        if (trainingConfig.LossFunction == LossFunction.Mse)
        {
            _lossFunction = new CpuMseLoss(Buffers, network);
        }
        else
        {
            _lossFunction = new CpuCrossEntropyLoss(Buffers, network);
        }
    }
    
    public NetworkConfig Network { get; }

    public NetworkAcceleratorBuffers Buffers { get; }

    public bool IsTraining => true;

    private readonly ILossFunction _lossFunction;

    private readonly TrainConfig _trainConfig;
    public void Dispose()
    {
        Buffers.Dispose();
        _lossFunction.Dispose();
        _optimizer.Dispose();
    }

    public void InitRandomParameters()
    {
        foreach (var networkLayer in _layers) networkLayer.FillRandom();
    }

    private List<float[]> Predict(List<float[]> batches)
    {
        var outputs = new List<float[]>();
        var batchSize = Buffers.BatchSize;
        var finalLayer = _layers[^1];

        // Divide the data into mini-batches
        for (var batchStart = 0; batchStart < batches.Count; batchStart += batchSize)
        {
            // Get the current batch
            var batch = batches.Skip(batchStart).Take(batchSize).ToList();
            var inputLayer = _layers[0];
            for (var i = 0; i < batch.Count; i++)
                Array.Copy(batch[i], 0, Buffers.Activations, i * Network.NetworkData.ActivationCount,
                    inputLayer.Config.NumInputs);

            foreach (var layer in _layers)
            {
                layer.Forward();
            }

            for (var i = 0; i < batch.Count; i++)
            {
                var output = new float[finalLayer.Config.NumOutputs];
                Array.Copy(Buffers.Activations, i * Network.NetworkData.ActivationCount + finalLayer.Config.ActivationOutputOffset, output, 0,
                    finalLayer.Config.NumOutputs);
                outputs.Add(output);
            }
        }

        return outputs;
    }

    private void Reset()
    {
        _optimizer.Reset();
    }

    private float TrainOnBatch(List<(float[] inputs, float[] expectedOutputs)> batch, float learningRate)
    {
        var inputLayer = _layers[0];
        
        for (var i = 0; i < batch.Count; i++)
            Array.Copy(batch[i].inputs, 0, Buffers.Activations, i * Network.NetworkData.ActivationCount,
                inputLayer.Config.NumInputs);
        
        foreach (var layer in _layers)
        {
            layer.Forward();
        }

        var loss = _lossFunction.Apply(batch);
        
        // Backward Pass
        for (var i = _layers.Length - 1; i >= 0; i--)
        {
            _layers[i].Backward();
        }
        
        _optimizer.Optimize(learningRate);

        return loss;
    }
    
    public void Train(List<(float[] imageData, float[] label)> data)
    {
        Console.WriteLine("Training network");
        Reset();

        var learningRateScheduler = LearningRateSchedulerFactory.Create(_trainConfig.Scheduler);
        var batchSize = Buffers.BatchSize;
        var totalBatches = (int)Math.Ceiling((double)data.Count / batchSize);

        var stopwatch = Stopwatch.StartNew();
        for (var epoch = 0; epoch < _trainConfig.Epochs; epoch++)
        {
            // Shuffle data in-place with Fisher-Yates for efficiency
            for (var i = data.Count - 1; i > 0; i--)
            {
                var j = Random.Shared.Next(i + 1);
                (data[i], data[j]) = (data[j], data[i]);
            }

            float epochError = 0;
            var sampleCount = 0;
            var learningRate = learningRateScheduler.GetLearningRate(epoch);
            for (var batch = 0; batch < totalBatches; batch++)
            {
                // Slice data for the current batch
                var batchData = data.GetRange(batch * batchSize, Math.Min(batchSize, data.Count - batch * batchSize));

                // Execute forward and backward pass on the accelerator
                var batchError = TrainOnBatch(batchData, learningRate);

                // Accumulate batch error and sample count
                epochError += batchError;
                sampleCount += batchData.Count;
            }

            var averageLoss = epochError / sampleCount;
            var elapsedTime = stopwatch.ElapsedMilliseconds;
            var samplesPerSec = Math.Round(sampleCount / stopwatch.Elapsed.TotalSeconds);

            Console.WriteLine(
                $"Epoch {epoch}, LR: {learningRate.ToSignificantFigures(3)} Average MSE: {averageLoss.ToSignificantFigures(3)}, Time: {elapsedTime}ms, {samplesPerSec}/s");
            stopwatch.Restart();
        }
    }

    public void Test(List<(float[] imageData, float[] label)> data,
        float threshold)
    {
        foreach (var layer in _layers)
        {
            layer.IsTraining = false;
        }
        
        Console.WriteLine("Testing network");

        var correct = 0;
        var incorrect = 0;
        var totalLabels = 0;
        var truePositives = new float[data.First().label.Length];
        var falsePositives = new float[data.First().label.Length];
        var falseNegatives = new float[data.First().label.Length];
        var trueNegatives = new float[data.First().label.Length];

        var shuffledData = data.OrderBy(x => Random.Shared.Next()).ToList();
        var stopwatch = Stopwatch.StartNew();

        // Get predictions from the model
        var predicted = Predict(shuffledData.Select(s => s.imageData).ToList());

        for (var index = 0; index < predicted.Count; index++)
        {
            var prediction = predicted[index];
            var expected = shuffledData[index].label;

            // Check if the full set of labels match for subset accuracy
            if (IsSubsetPredictionCorrect(expected, prediction, threshold))
                correct++;
            else
                incorrect++;

            // Update confusion matrix statistics for each label
            for (var i = 0; i < expected.Length; i++)
            {
                var expectedLabel = expected[i] > threshold;
                var predictedLabel = prediction[i] > threshold;

                if (expectedLabel && predictedLabel)
                    truePositives[i]++;
                if (!expectedLabel && predictedLabel)
                    falsePositives[i]++;
                if (expectedLabel && !predictedLabel)
                    falseNegatives[i]++;
                if (!expectedLabel && !predictedLabel)
                    trueNegatives[i]++;
            }

            totalLabels += expected.Length;
        }

        // Precision, Recall, F1-Score for each label
        for (var i = 0; i < truePositives.Length; i++)
        {
            var precision = Precision(truePositives[i], falsePositives[i]);
            var recall = Recall(truePositives[i], falseNegatives[i]);
            var f1Score = F1Score(precision, recall);

            Console.WriteLine($"Class {i}: Precision: {precision}, Recall: {recall}, F1-Score: {f1Score}");
        }

        var total = correct + incorrect;
        var accuracy = correct / (float)total * 100;
        Console.WriteLine("Subset Accuracy: {0}/{1} ({2}%)", correct, total, Math.Round(accuracy, 2));
        var elapsedTime = stopwatch.ElapsedMilliseconds;
        var throughput = total / (float)stopwatch.Elapsed.TotalSeconds;
        Console.WriteLine($"Time: {elapsedTime}ms, Throughput: {throughput}/s");
    }

    // Helper function for subset accuracy (strict comparison)
    private static bool IsSubsetPredictionCorrect(float[] expected, float[] actual, float threshold)
    {
        // Check if all labels for a sample are predicted correctly
        for (var i = 0; i < expected.Length; i++)
            if (expected[i] > threshold != actual[i] > threshold)
                return false; // If any label is incorrect, return false
        return true;
    }

    // Precision (true positives / (true positives + false positives))
    private static float Precision(float truePositives, float falsePositives)
    {
        return truePositives + falsePositives == 0 ? 0 : truePositives / (truePositives + falsePositives);
    }

    // Recall (true positives / (true positives + false negatives))
    private static float Recall(float truePositives, float falseNegatives)
    {
        return truePositives + falseNegatives == 0 ? 0 : truePositives / (truePositives + falseNegatives);
    }

    // F1 Score (harmonic mean of Precision and Recall)
    private static float F1Score(float precision, float recall)
    {
        return precision + recall == 0 ? 0 : 2 * (precision * recall) / (precision + recall);
    }
    
    #region Io

    public void SaveParametersToDisk(string filePath)
    {
        using var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write);
        using var writer = new BinaryWriter(stream);
        // Write the number of arrays to allow easy deserialization
        writer.Write(Network.NetworkData.ParameterCount);
        foreach (var value in Buffers.Parameters) writer.Write(value);
    }

    public void ReadParametersFromDisk(string filePath)
    {
        using var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
        using var reader = new BinaryReader(stream);
        // Read the number of arrays
        var length = reader.ReadInt32();

        for (var j = 0; j < length; j++) Buffers.Parameters[j] = reader.ReadSingle();
    }

    public float[] GetParameters()
    {
        var parameters = new float[Network.NetworkData.ParameterCount];
        Array.Copy(Buffers.Parameters, 0, parameters, 0, parameters.Length);
        return parameters;
    }

    public void LoadParameters(float[] parameters)
    {
        Array.Copy(parameters, 0, Buffers.Parameters,0, parameters.Length);
    }

    #endregion
}