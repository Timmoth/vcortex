using System.Diagnostics;
using vcortex.cpu.Layers;
using vcortex.cpu.Optimizers;
using vcortex.gpu.Optimizers;
using vcortex.Layers;
using vcortex.LearningRate;
using vcortex.Network;
using vcortex.Optimizers;
using vcortex.Training;

namespace vcortex.cpu;

public class NetworkTrainer : INetworkAgent
{
    private readonly float[] _flattenedExpectedOutputs;

    private readonly float[] _flattenedInputs;
    private readonly ILayer[] _layers;


    private readonly IOptimizer _optimizer;

    public NetworkTrainer(NetworkConfig network, LossFunction lossFunction, OptimizerConfig optimizer, int batchSize)
    {
        Network = network;

        _layers = network.Layers.Select(CpuLayerFactory.Create).ToArray();
        _optimizer = CpuOptimizerFactory.Create(optimizer);

        Buffers = new NetworkAcceleratorBuffers(network, batchSize);

        var inputLayer = _layers[0];
        var inputCount = inputLayer.Config.NumInputs * Buffers.BatchSize;
        _flattenedInputs = new float[inputCount];

        var outputLayer = _layers[^1];
        var outputCount = outputLayer.Config.NumOutputs * Buffers.BatchSize;
        _flattenedExpectedOutputs = new float[outputCount];
    }
    public NetworkConfig Network { get; }

    public NetworkAcceleratorBuffers Buffers { get; }

    public bool IsTraining => true;


    public void Dispose()
    {
        Buffers.Dispose();
    }

    public void InitRandomWeights()
    {
        foreach (var networkLayer in _layers) networkLayer.FillRandom();
    }

    private List<float[]> Predict(List<float[]> batchs)
    {
        var outputs = new List<float[]>();
        var batchSize = Buffers.BatchSize;
        var finalLayer = _layers[^1];

        // Divide the data into mini-batches
        for (var batchStart = 0; batchStart < batchs.Count; batchStart += batchSize)
        {
            // Get the current batch
            var batch = batchs.Skip(batchStart).Take(batchSize).ToList();
            var inputLayer = _layers[0];
            for (var i = 0; i < batch.Count; i++)
                Array.Copy(batch[i], 0, _flattenedInputs, i * inputLayer.Config.NumInputs,
                    inputLayer.Config.NumInputs);

            foreach (var layer in _layers)
            {
                layer.Forward();
            }

            for (var i = 0; i < batch.Count; i++)
            {
                var output = new float[finalLayer.Config.NumOutputs];
                outputs.Add(output);
            }
        }

        return outputs;
    }

    private void Reset()
    {
        _optimizer.Reset();
    }

    private float Train(List<(float[] inputs, float[] expectedOutputs)> batch, float learningRate)
    {
        var stopwatch = Stopwatch.StartNew();

        var inputLayer = _layers[0];
        var outputLayer = _layers[^1];

        for (var i = 0; i < batch.Count; i++)
            Array.Copy(batch[i].inputs, 0, _flattenedInputs, i * inputLayer.Config.NumInputs,
                inputLayer.Config.NumInputs);

     
        foreach (var layer in _layers)
        {
            layer.Forward();
        }

        var finalLayer = _layers[^1];

        for (var i = 0; i < batch.Count; i++)
            Array.Copy(batch[i].expectedOutputs, 0, _flattenedExpectedOutputs, i * outputLayer.Config.NumOutputs,
                outputLayer.Config.NumOutputs);

        var sampleError = _flattenedExpectedOutputs.Sum();

        // Backward Pass
        for (var i = _layers.Length - 1; i >= 0; i--)
        {
            _layers[i].Backward();
        }

        _optimizer.Optimize(learningRate);

        //Console.WriteLine($" final sync: {stopwatch.ElapsedMilliseconds}ms");
        stopwatch.Restart();
        return sampleError / finalLayer.Config.NumOutputs;
    }


    public void TrainAccelerated(List<(float[] imageData, float[] label)> data, TrainConfig trainConfig)
    {
        Console.WriteLine("Training network");
        Reset();

        var learningRateScheduler = LearningRateSchedulerFactory.Create(trainConfig.Scheduler);
        var batchSize = Buffers.BatchSize;
        var totalBatches = (int)Math.Ceiling((double)data.Count / batchSize);

        var stopwatch = Stopwatch.StartNew();
        for (var epoch = 0; epoch < trainConfig.Epochs; epoch++)
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
                var batchError = Train(batchData, learningRate);

                // Accumulate batch error and sample count
                epochError += batchError;
                sampleCount += batchData.Count;
            }

            var averageMSE = epochError / sampleCount;
            var elapsedTime = stopwatch.ElapsedMilliseconds;
            var samplesPerSec = Math.Round(sampleCount / stopwatch.Elapsed.TotalSeconds);

            Console.WriteLine(
                $"Epoch {epoch}, LR: {learningRate.ToSignificantFigures(3)} Average MSE: {averageMSE.ToSignificantFigures(3)}, Time: {elapsedTime}ms, {samplesPerSec}/s");
            stopwatch.Restart();
        }
    }

    public void Test(List<(float[] imageData, float[] label)> data,
        float threshold)
    {
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
}