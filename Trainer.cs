using System.Diagnostics;
using ILGPU.Algorithms;
using vcortex.Accelerated;

namespace vcortex;

public static class Trainer
{
    public static void TrainAccelerated(NetworkAccelerator accelerator, List<(float[] imageData, float[] label)> data,
        int epochs)
    {
        Console.WriteLine("Training network");
        accelerator.Network.NetworkData = accelerator.Network.NetworkData.ResetTimestep();

        // Forward Pass
        var batchSize = accelerator.Network.NetworkData.BatchSize;
        for (var epoch = 0; epoch < epochs; epoch++)
        {
            // Shuffle the data at the beginning of each epoch
            var shuffledData = data.OrderBy(x => Random.Shared.Next()).ToList();
            var stopwatch = Stopwatch.StartNew();

            float epochError = 0;
            var sampleCount = 0;

            // Divide the data into mini-batches
            for (var batchStart = 0; batchStart < shuffledData.Count; batchStart += batchSize)
            {
                // Get the current batch
                var currentBatch = shuffledData.Skip(batchStart).Take(batchSize).ToList();

                // Perform forward and backward passes and get the batch error
                var batchError = accelerator.Train(currentBatch);

                // Accumulate batch error
                epochError += batchError;
                sampleCount += currentBatch.Count;
            }

            // Calculate and report the average MSE for the epoch
            var averageMSE = epochError / sampleCount;
            Console.WriteLine(
                $"Epoch {epoch}, Average MSE: {averageMSE:F4}, Time: {stopwatch.ElapsedMilliseconds}ms, {Math.Round(sampleCount / stopwatch.Elapsed.TotalSeconds)}/s");
        }
    }

    public static void Test(NetworkAccelerator accelerator, List<(float[] imageData, float[] label)> data,
        float threshold)
    {
        Console.WriteLine("Testing network");
        var correct = 0;
        var incorrect = 0;

        var shuffledData = data.OrderBy(x => Random.Shared.Next()).ToList();
        var stopwatch = Stopwatch.StartNew();

        // Train using shuffled data and accumulate error
        var predicted = accelerator.Predict(shuffledData.Select(s => s.imageData).ToList());

        for (var index = 0; index < predicted.Count; index++)
        {
            var prediction = predicted[index];
            if (IsMultiLabelPredictionCorrect(shuffledData[index].label, prediction, threshold))
                correct++;
            else
                incorrect++;
        }

        var total = correct + incorrect;
        Console.WriteLine("Correct: {0}/{1} ({2}%) in: {3}ms {4}/s", correct, total,
            XMath.Round(correct / (float)total * 100, 2), stopwatch.ElapsedMilliseconds,
            XMath.Round(total / (float)stopwatch.Elapsed.TotalSeconds));
    }

    private static bool IsMultiLabelPredictionCorrect(float[] expected, float[] actual, float threshold)
    {
        // Iterate over each class and check if both expected and actual values are above or below the threshold
        for (var i = 0; i < expected.Length; i++)
        {
            var expectedLabel = expected[i] > threshold;
            var predictedLabel = actual[i] > threshold;

            // If there's a mismatch for any class, return false
            if (expectedLabel != predictedLabel) return false;
        }

        return true; // All labels match
    }
}