using System.Diagnostics;
using vcortex.Layers;
using vcortex.Optimizers;
using vcortex.Training;

namespace vcortex.Network;

public abstract class NetworkTrainerBase : INetworkTrainerAgent
{
    protected abstract ILayer[] Layers { get; }
    protected abstract TrainConfig TrainingConfig { get; }
    protected abstract IOptimizer Optimizer { get; }
    protected abstract ILossFunction LossFunction { get; }
    protected abstract void Reset();
    protected abstract List<float[]> Predict(List<float[]> batches);
    public abstract void Dispose();
    public abstract void SaveParametersToDisk(string filePath);
    public abstract void ReadParametersFromDisk(string filePath);
    public abstract float[] GetParameters();
    public abstract void LoadParameters(float[] parameters);
    public void InitRandomParameters()
    {
        foreach (var networkLayer in Layers) networkLayer.FillRandom();
    }

    protected abstract void InitBatch(List<(float[] inputs, float[] expectedOutputs)> batch);
    private float TrainOnBatch(List<(float[] inputs, float[] expectedOutputs)> batch, float learningRate)
    {
        InitBatch(batch);
        foreach (var layer in Layers)
        {
            layer.Forward();
        }
        
        var loss = LossFunction.Apply(batch);
        for (var i = Layers.Length - 1; i >= 0; i--)
        {
            Layers[i].Backward();
        }

        Optimizer.Optimize(learningRate);
        return loss;
    }
    
    public void Train(List<(float[] imageData, float[] label)> data)
    {
        Console.WriteLine($"Training network");
        Reset();

        var learningRateScheduler = LearningRateSchedulerFactory.Create(TrainingConfig.Scheduler);
        var batchSize = TrainingConfig.BatchSize;
        var totalBatches = (int)Math.Ceiling((double)data.Count / batchSize);

        var stopwatch = Stopwatch.StartNew();
        for (var epoch = 0; epoch < TrainingConfig.Epochs; epoch++)
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
                $"Epoch {epoch}, Lr: {learningRate.ToSignificantFigures(3)} Loss: {averageLoss.ToSignificantFigures(3)}, Time: {elapsedTime}ms, {samplesPerSec}/s");
            stopwatch.Restart();
        }
    }

    public void Test(List<(float[] imageData, float[] label)> data,
        float threshold)
    {
        foreach (var layer in Layers)
        {
            layer.IsTraining = false;
        }
        
        Console.WriteLine($"Testing network");

        var correct = 0;
        var incorrect = 0;
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