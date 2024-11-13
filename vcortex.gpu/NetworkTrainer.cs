using System.Diagnostics;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using vcortex.Core;
using vcortex.Core.Layers;
using vcortex.Core.Optimizers;
using vcortex.gpu.Layers;
using vcortex.gpu.Optimizers;
using vcortex.LearningRate;

namespace vcortex.gpu;

public class NetworkTrainer : INetworkAgent
{
    private readonly Context _context;
    private readonly float[] _flattenedExpectedOutputs;

    private readonly float[] _flattenedInputs;
    private readonly ILayer[] _layers;

    private readonly Action<Index1D, NetworkData, ArrayView<float>, ArrayView<float>, int> _loadInputsKernel;

    private readonly Action<Index1D, NetworkData, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>
        _lossFunctionKernel;

    private readonly IOptimizer _optimizer;

    public NetworkTrainer(NetworkConfig network, LossFunction lossFunction, OptimizerConfig optimizer, int batchSize)
    {
        Network = network;

        _layers = network.Layers.Select(GpuLayerFactory.Create).ToArray();
        _optimizer = GpuOptimizerFactory.Create(optimizer);
        _context = Context.Create(b => { b.Default().EnableAlgorithms().Math(MathMode.Fast); });

        var useCuda = true;
        if (useCuda)
        {
            foreach (var device in _context.GetCudaDevices()) Console.WriteLine(device.Name + " " + device.DeviceId);

            Accelerator = _context.CreateCudaAccelerator(0);
        }
        else
        {
            _context = Context.Create(b => { b.Default().EnableAlgorithms().CPU(); });

            Accelerator = _context.CreateCPUAccelerator(0);
        }

        Buffers = new NetworkAcceleratorBuffers(Accelerator, network, batchSize);

        foreach (var layer in _layers) layer.CompileKernels(this);

        _optimizer.Compile(this);

        _loadInputsKernel =
            Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, ArrayView<float>, ArrayView<float>, int>(
                    LoadInputs);

        _lossFunctionKernel =
            Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, int, int, int>(
                    lossFunction == LossFunction.Mse ? MSE : CrossEntropyLoss);

        var inputLayer = _layers[0];
        var inputCount = inputLayer.Config.NumInputs * Buffers.BatchSize;
        _flattenedInputs = new float[inputCount];

        var outputLayer = _layers[^1];
        var outputCount = outputLayer.Config.NumOutputs * Buffers.BatchSize;
        _flattenedExpectedOutputs = new float[outputCount];
    }

    public Accelerator Accelerator { get; }

    public NetworkConfig Network { get; }

    public NetworkAcceleratorBuffers Buffers { get; }

    public bool IsTraining => true;


    public void Dispose()
    {
        Accelerator.Dispose();
        _context.Dispose();
        Buffers.Dispose();
        _optimizer.Dispose();
    }

    public void InitRandomWeights()
    {
        foreach (var networkLayer in _layers) networkLayer.FillRandom(this);
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

            Buffers.Inputs.View.CopyFromCPU(_flattenedInputs);
            _loadInputsKernel(_flattenedInputs.Length, Network.NetworkData, Buffers.Inputs.View,
                Buffers.Activations.View, inputLayer.Config.NumInputs);

            foreach (var layer in _layers)
            {
                layer.Forward(this);
                Accelerator.Synchronize();
            }

            for (var i = 0; i < batch.Count; i++)
            {
                var output = new float[finalLayer.Config.NumOutputs];
                Buffers.Activations.View
                    .SubView(Network.NetworkData.ActivationCount * i + finalLayer.Config.ActivationOutputOffset,
                        finalLayer.Config.NumOutputs).CopyToCPU(output);
                outputs.Add(output);
            }
        }

        return outputs;
    }


    public static void LoadInputs(
        Index1D index,
        NetworkData networkData,
        ArrayView<float> inputs,
        ArrayView<float> activations, int numInputs)
    {
        // Number of samples in the batch
        var batchIndex = index / numInputs;
        var inputIndex = index % numInputs;

        activations[networkData.ActivationCount * batchIndex + inputIndex] =
            inputs[numInputs * batchIndex + inputIndex];
    }

    public static void MSE(
        Index1D index,
        NetworkData networkData,
        ArrayView<float> outputs,
        ArrayView<float> activations,
        ArrayView<float> errors, int numOutputs, int activationOutputOffset, int nextLayerErrorOffset)
    {
        // Number of samples in the batch
        var batchIndex = index / numOutputs;
        var outputIndex = index % numOutputs;
        var activationOutputIndex = batchIndex * networkData.ActivationCount + activationOutputOffset + outputIndex;
        var nextErrorOffset = batchIndex * networkData.ActivationCount + nextLayerErrorOffset;

        var expected = outputs[numOutputs * batchIndex + outputIndex];
        var actual = activations[activationOutputIndex];
        var error = actual - expected;

        outputs[numOutputs * batchIndex + outputIndex] = error * error;
        errors[nextErrorOffset + outputIndex] = error;
    }

    public static void CrossEntropyLoss(
        Index1D index,
        NetworkData networkData,
        ArrayView<float> outputs,
        ArrayView<float> activations,
        ArrayView<float> errors, int numOutputs, int activationOutputOffset, int nextLayerErrorOffset)
    {
        // Number of samples in the batch
        var batchIndex = index / numOutputs;
        var outputIndex = index % numOutputs;
        var activationOutputIndex = batchIndex * networkData.ActivationCount + activationOutputOffset + outputIndex;
        var nextErrorOffset = batchIndex * networkData.ActivationCount + nextLayerErrorOffset;

        // Get the expected and actual values
        var expected = outputs[numOutputs * batchIndex + outputIndex];
        var actual = activations[activationOutputIndex];

        // Compute Cross-Entropy loss for each output (assuming outputs are one-hot encoded)
        // We use a small epsilon to prevent log(0)
        var epsilon = 1e-15f;
        var logProb = MathF.Max(actual, epsilon); // Log of the predicted probability (softmax output)

        // Compute the loss for the current sample
        var loss = -expected * MathF.Log(logProb);

        // Store the loss in the outputs array (you could sum these later for the full batch loss)
        outputs[numOutputs * batchIndex + outputIndex] = loss;

        // Calculate the gradient of the loss w.r.t. the predicted probability (backpropagation)
        // Derivative of cross-entropy loss with softmax is: p - y
        var gradient = actual - expected;

        // Store the gradient in the errors array
        errors[nextErrorOffset + outputIndex] = gradient;
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

        Buffers.Inputs.View.CopyFromCPU(_flattenedInputs);
        _loadInputsKernel(_flattenedInputs.Length, Network.NetworkData, Buffers.Inputs.View,
            Buffers.Activations.View, inputLayer.Config.NumInputs);
        foreach (var layer in _layers)
        {
            layer.Forward(this);
            Accelerator.Synchronize();
        }

        var finalLayer = _layers[^1];

        for (var i = 0; i < batch.Count; i++)
            Array.Copy(batch[i].expectedOutputs, 0, _flattenedExpectedOutputs, i * outputLayer.Config.NumOutputs,
                outputLayer.Config.NumOutputs);

        Buffers.Errors.View.MemSetToZero();
        Buffers.Outputs.View.CopyFromCPU(_flattenedExpectedOutputs);
        _lossFunctionKernel(Buffers.BatchSize * outputLayer.Config.NumOutputs, Network.NetworkData,
            Buffers.Outputs.View, Buffers.Activations.View, Buffers.Errors.View, outputLayer.Config.NumOutputs,
            outputLayer.Config.ActivationOutputOffset, outputLayer.Config.NextLayerErrorOffset);
        Buffers.Outputs.View.CopyToCPU(_flattenedExpectedOutputs);
        var sampleError = _flattenedExpectedOutputs.Sum();

        // Backward Pass
        for (var i = _layers.Length - 1; i >= 0; i--)
        {
            _layers[i].Backward(this);
            Accelerator.Synchronize();
        }

        _optimizer.Optimize(Network.NetworkData, Buffers, learningRate);
        Accelerator.Synchronize();

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
        Console.WriteLine("Subset Accuracy: {0}/{1} ({2}%)", correct, total, XMath.Round(accuracy, 2));
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