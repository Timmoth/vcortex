using System.Diagnostics;
using System.Net;
using System.Reflection.Emit;
using ILGPU.Algorithms;
using ILGPU.Util;
using vcortex.Accelerated;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace vcortex;

public static class Trainer
{
    public static void TrainAccelerated(NetworkAccelerator accelerator, List<(float[] imageData, float[] label)> data, int epochs,
        float learningRate)
    {
        Console.WriteLine("Training network");
        accelerator.Network.NetworkData.LearningRate = learningRate;

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

    public static void TrainBatched(Network network, List<(float[] imageData, float[] label)> data, int epochs,
        float learningRate, int batchSize)
    {
        Console.WriteLine("Training network");

        var activations = new float[batchSize][];
        var errors = new float[batchSize][];
        var gradients = new float[batchSize][];

        for (var i = 0; i < batchSize; i++)
        {
            activations[i] = new float[network.ActivationCount];
            errors[i] = new float[network.ActivationCount]; 
            gradients[i] = new float[network.GradientCount];
        }

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
                var batchError = network.Train(currentBatch, activations, errors, gradients, learningRate);

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


    public static void TrainSequential(Network network, List<(float[] imageData, float[] label)> data, int epochs,
        float learningRate)
    {
        Console.WriteLine("Training network");

        var activations = new float[network.ActivationCount];
        var errors = new float[network.ActivationCount];

        var gradients = new float[network.GradientCount];

        // Forward Pass
        for (var i = 0; i < epochs; i++)
        {
            var shuffledData = data.OrderBy(x => Random.Shared.Next()).ToList();
            var stopwatch = Stopwatch.StartNew();

            float epochError = 0;
            var sampleCount = 0;

            foreach (var (imageData, label) in shuffledData)
            {
                var sampleError = network.Train(imageData, label, activations, errors, gradients, learningRate);
                epochError += sampleError;
                sampleCount++;

                if (sampleCount > 1000)
                {
                    break;
                }
            }

            // Calculate average MSE for the epoch
            var averageMSE = epochError / sampleCount;
            Console.WriteLine($"Epoch {i}, Average MSE: {averageMSE:F4}, Time: {stopwatch.ElapsedMilliseconds}ms");
        }
    }

    public static void Test(NetworkAccelerator accelerator, List<(float[] imageData, float[] label)> data, float threshold)
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
            XMath.Round(correct / (float)total * 100, 2), stopwatch.ElapsedMilliseconds, XMath.Round(total / (float)stopwatch.Elapsed.TotalSeconds));
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