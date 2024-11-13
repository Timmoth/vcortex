using System.Diagnostics;
using ILGPU.Algorithms;
using vcortex.Accelerated;

namespace vcortex;

public static class Trainer
{
    public static void TrainAccelerated(NetworkAccelerator accelerator, List<(float[] imageData, float[] label)> data, int epochs)
    {
        Console.WriteLine("Training network");
        accelerator.Network.NetworkData = accelerator.Network.NetworkData.ResetTimestep();

        var batchSize = accelerator.Network.NetworkData.BatchSize;
        var totalBatches = (int)Math.Ceiling((double)data.Count / batchSize);

        var stopwatch = Stopwatch.StartNew();
        for (var epoch = 0; epoch < epochs; epoch++)
        {
            // Shuffle data in-place with Fisher-Yates for efficiency
            for (int i = data.Count - 1; i > 0; i--)
            {
                int j = Random.Shared.Next(i + 1);
                (data[i], data[j]) = (data[j], data[i]);
            }

            float epochError = 0;
            int sampleCount = 0;

            for (var batch = 0; batch < totalBatches; batch++)
            {
                // Slice data for the current batch
                var batchData = data.GetRange(batch * batchSize, Math.Min(batchSize, data.Count - batch * batchSize));

                // Execute forward and backward pass on the accelerator
                var batchError = accelerator.Train(batchData);

                // Accumulate batch error and sample count
                epochError += batchError;
                sampleCount += batchData.Count;
            }

            var averageMSE = epochError / sampleCount;
            var elapsedTime = stopwatch.ElapsedMilliseconds;
            var samplesPerSec = Math.Round(sampleCount / stopwatch.Elapsed.TotalSeconds);

            Console.WriteLine($"Epoch {epoch}, Average MSE: {averageMSE:F4}, Time: {elapsedTime}ms, {samplesPerSec}/s");
            stopwatch.Restart();
        }
    }


    public static void Test(NetworkAccelerator accelerator, List<(float[] imageData, float[] label)> data,
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
        var predicted = accelerator.Predict(shuffledData.Select(s => s.imageData).ToList());

        for (var index = 0; index < predicted.Count; index++)
        {
            var prediction = predicted[index];
            var expected = shuffledData[index].label;

            // Check if the full set of labels match for subset accuracy
            if (IsSubsetPredictionCorrect(expected, prediction, threshold))
            {
                correct++;
            }
            else
            {
                incorrect++;
            }

            // Update confusion matrix statistics for each label
            for (var i = 0; i < expected.Length; i++)
            {
                bool expectedLabel = expected[i] > threshold;
                bool predictedLabel = prediction[i] > threshold;

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

        var total = correct + incorrect;
        var accuracy = correct / (float)total * 100;
        Console.WriteLine("Subset Accuracy: {0}/{1} ({2}%)", correct, total, XMath.Round(accuracy, 2));

        // Precision, Recall, F1-Score for each label
        for (int i = 0; i < truePositives.Length; i++)
        {
            float precision = Precision(truePositives[i], falsePositives[i]);
            float recall = Recall(truePositives[i], falseNegatives[i]);
            float f1Score = F1Score(precision, recall);

            Console.WriteLine($"Class {i}: Precision: {precision}, Recall: {recall}, F1-Score: {f1Score}");
        }

        var elapsedTime = stopwatch.ElapsedMilliseconds;
        var throughput = total / (float)stopwatch.Elapsed.TotalSeconds;
        Console.WriteLine($"Time: {elapsedTime}ms, Throughput: {throughput}/s");
    }

    // Helper function for subset accuracy (strict comparison)
    private static bool IsSubsetPredictionCorrect(float[] expected, float[] actual, float threshold)
    {
        // Check if all labels for a sample are predicted correctly
        for (var i = 0; i < expected.Length; i++)
        {
            if ((expected[i] > threshold) != (actual[i] > threshold))
                return false; // If any label is incorrect, return false
        }
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