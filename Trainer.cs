using System.Diagnostics;

namespace vcortex;

public static class Trainer
{
    public static void TrainBatched(Network network, List<(float[] imageData, float[] label)> data, int epochs,
        float learningRate, int batchSize)
    {
        Console.WriteLine("Training network");

        var networkData = new NetworkData[batchSize];
        for (var i = 0; i < batchSize; i++) networkData[i] = new NetworkData(network._layers);

        var gradients = new float[network._layers.Length][][];
        for (var i = 0; i < network._layers.Length; i++)
        {
            gradients[i] = new float[batchSize][];
            var gradientCount = network._layers[i].GradientCount;
            for (var j = 0; j < batchSize; j++) gradients[i][j] = new float[gradientCount];
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
                var batchError = network.Train(currentBatch, networkData, gradients, learningRate);

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

        var networkData = new NetworkData(network._layers);
        var gradients = new float[network._layers.Length][];

        // Forward Pass
        gradients[0] = new float[network._layers[0].GradientCount];

        for (var i = 1; i < network._layers.Length; i++) gradients[i] = new float[network._layers[i].GradientCount];

        for (var i = 0; i < epochs; i++)
        {
            var shuffledData = data.OrderBy(x => Random.Shared.Next()).ToList();
            var stopwatch = Stopwatch.StartNew();

            float epochError = 0;
            var sampleCount = 0;

            foreach (var (imageData, label) in shuffledData)
            {
                var sampleError = network.Train(imageData, label, networkData, gradients, learningRate);
                epochError += sampleError;
                sampleCount++;
            }

            // Calculate average MSE for the epoch
            var averageMSE = epochError / sampleCount;
            Console.WriteLine($"Epoch {i}, Average MSE: {averageMSE:F4}, Time: {stopwatch.ElapsedMilliseconds}ms");
        }
    }

    public static void Test(Network network, List<(float[] imageData, float[] label)> data, float threshold)
    {
        Console.WriteLine("Testing network");
        var correct = 0;
        var incorrect = 0;

        var shuffledData = data.OrderBy(x => Random.Shared.Next()).ToList();
        var stopwatch = Stopwatch.StartNew();

        // Train using shuffled data and accumulate error
        foreach (var (imageData, label) in shuffledData)
        {
            var predicted = network.Predict(imageData);

            if (IsMultiLabelPredictionCorrect(label, predicted, threshold))
                correct++;
            else
                //Console.WriteLine("Incorrect: {0} - {1}", string.Join(", ", label.Select(v => Math.Round(v, 2))), string.Join(", ", predicted.Select(v => Math.Round(v, 2))));
                incorrect++;
        }

        var total = correct + incorrect;
        Console.WriteLine($"Correct: {{0}}/{{1}} ({{2}}%) in: {3}ms", correct, total,
            Math.Round(correct / (float)total * 100, 2), stopwatch.ElapsedMilliseconds);
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