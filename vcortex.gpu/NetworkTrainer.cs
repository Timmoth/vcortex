using System.Diagnostics;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using vcortex.Core;
using vcortex.Core.Optimizers;
using vcortex.gpu.Optimizers;
using vcortex.LearningRate;

namespace vcortex.gpu;

public class NetworkTrainer : INetworkAgent
{
    private readonly float[] _flattenedExpectedOutputs;

    private readonly float[] _flattenedInputs;
    private readonly Accelerator _accelerator;
    private readonly Context _context;
    private readonly NetworkAcceleratorBuffers _buffers;
    private readonly IOptimizer _optimizer;
    private readonly Network _network;

    public Accelerator Accelerator => _accelerator;
    public Network Network => _network;
    public NetworkAcceleratorBuffers Buffers => _buffers;
    public bool IsTraining => true;
    
    public NetworkTrainer(Network network, LossFunction lossFunction, OptimizerConfig optimizer, int batchSize)
    {
        _network = network;
        _optimizer = GpuOptimizerFactory.Create(optimizer);
        _context = Context.Create(b => { b.Default().EnableAlgorithms().Math(MathMode.Fast); });

        var useCuda = true;
        if (useCuda)
        {
            foreach (var device in _context.GetCudaDevices()) Console.WriteLine(device.Name + " " + device.DeviceId);

            _accelerator = _context.CreateCudaAccelerator(0);
        }
        else
        {
            _context = Context.Create(b => { b.Default().EnableAlgorithms().CPU(); });

            _accelerator = _context.CreateCPUAccelerator(0);
        }

        _buffers = new NetworkAcceleratorBuffers(_accelerator, network, batchSize);

        foreach (var layer in network._layers) layer.CompileKernels(this);

        _optimizer.Compile(this);
        
        _loadInputsKernel =
            _accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>>(
                    LoadInputs);

        _lossFunctionKernel =
            _accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>>(
                    lossFunction == LossFunction.Mse ? MSE : CrossEntropyLoss);

        var inputLayer = Network._layers[0];
        var inputCount = inputLayer.LayerData.NumInputs * _buffers.BatchSize;
        _flattenedInputs = new float[inputCount];

        var outputLayer = Network._layers[^1];
        var outputCount = outputLayer.LayerData.NumOutputs * _buffers.BatchSize;
        _flattenedExpectedOutputs = new float[outputCount];
    }

    private readonly Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>> _loadInputsKernel;

    private readonly Action<Index1D, NetworkData, LayerData, ArrayView<float>, ArrayView<float>, ArrayView<float>>
        _lossFunctionKernel;
    

    public void Dispose()
    {
        _accelerator.Dispose();
        _context.Dispose();
        _buffers.Dispose();
        _optimizer.Dispose();
    }

    public void InitRandomWeights()
    {
        foreach (var networkLayer in _network._layers) networkLayer.FillRandom(this);
    }

    private List<float[]> Predict(List<float[]> batchs)
    {
        var outputs = new List<float[]>();
        var batchSize = _buffers.BatchSize;
        var finalLayer = _network._layers[^1];

        // Divide the data into mini-batches
        for (var batchStart = 0; batchStart < batchs.Count; batchStart += batchSize)
        {
            // Get the current batch
            var batch = batchs.Skip(batchStart).Take(batchSize).ToList();
            var inputLayer = _network._layers[0];
            for (var i = 0; i < batch.Count; i++)
                Array.Copy(batch[i], 0, _flattenedInputs, i * inputLayer.LayerData.NumInputs,
                    inputLayer.LayerData.NumInputs);
            
            _buffers.Inputs.View.CopyFromCPU(_flattenedInputs);
            _loadInputsKernel(_flattenedInputs.Length, Network.NetworkData, inputLayer.LayerData, _buffers.Inputs.View,
                _buffers.Activations.View);
            
            foreach (var layer in _network._layers)
            {
                layer.Forward(this);
                _accelerator.Synchronize();
            }
            
            for (var i = 0; i < batch.Count; i++)
            {
                var output = new float[finalLayer.NumOutputs];
                _buffers.Activations.View
                    .SubView(Network.NetworkData.ActivationCount * i + finalLayer.ActivationOutputOffset,
                        finalLayer.NumOutputs).CopyToCPU(output);
                outputs.Add(output);
            }
        }

        return outputs;
    }


    public static void LoadInputs(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> inputs,
        ArrayView<float> activations)
    {
        // Number of samples in the batch
        var batchIndex = index / layerData.NumInputs;
        var inputIndex = index % layerData.NumInputs;

        activations[networkData.ActivationCount * batchIndex + inputIndex] =
            inputs[layerData.NumInputs * batchIndex + inputIndex];
    }

    public static void MSE(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> outputs,
        ArrayView<float> activations,
        ArrayView<float> errors)
    {
        // Number of samples in the batch
        var batchIndex = index / layerData.NumOutputs;
        var outputIndex = index % layerData.NumOutputs;
        var activationOutputOffset = batchIndex * networkData.ActivationCount + layerData.ActivationOutputOffset;
        var nextErrorOffset = batchIndex * networkData.ActivationCount + layerData.NextLayerErrorOffset;

        var expected = outputs[layerData.NumOutputs * batchIndex + outputIndex];
        var actual = activations[activationOutputOffset + outputIndex];
        var error = actual - expected;

        outputs[layerData.NumOutputs * batchIndex + outputIndex] = error * error;
        errors[nextErrorOffset + outputIndex] = error;
    }

    public static void CrossEntropyLoss(
        Index1D index,
        NetworkData networkData,
        LayerData layerData,
        ArrayView<float> outputs,
        ArrayView<float> activations,
        ArrayView<float> errors)
    {
        // Number of samples in the batch
        var batchIndex = index / layerData.NumOutputs;
        var outputIndex = index % layerData.NumOutputs;
        var activationOutputOffset = batchIndex * networkData.ActivationCount + layerData.ActivationOutputOffset;
        var nextErrorOffset = batchIndex * networkData.ActivationCount + layerData.NextLayerErrorOffset;

        // Get the expected and actual values
        var expected = outputs[layerData.NumOutputs * batchIndex + outputIndex];
        var actual = activations[activationOutputOffset + outputIndex];

        // Compute Cross-Entropy loss for each output (assuming outputs are one-hot encoded)
        // We use a small epsilon to prevent log(0)
        float epsilon = 1e-15f;
        var logProb = MathF.Max(actual, epsilon); // Log of the predicted probability (softmax output)

        // Compute the loss for the current sample
        var loss = -expected * MathF.Log(logProb);

        // Store the loss in the outputs array (you could sum these later for the full batch loss)
        outputs[layerData.NumOutputs * batchIndex + outputIndex] = loss;

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
        
        var inputLayer = Network._layers[0];
        var outputLayer = Network._layers[^1];

        for (var i = 0; i < batch.Count; i++)
            Array.Copy(batch[i].inputs, 0, _flattenedInputs, i * inputLayer.LayerData.NumInputs,
                inputLayer.LayerData.NumInputs);

        _buffers.Inputs.View.CopyFromCPU(_flattenedInputs);
        _loadInputsKernel(_flattenedInputs.Length, _network.NetworkData, inputLayer.LayerData, _buffers.Inputs.View,
            _buffers.Activations.View);
        foreach (var layer in Network._layers)
        {
            layer.Forward(this);
            _accelerator.Synchronize();
        }

        var finalLayer = _network._layers[^1];

        for (var i = 0; i < batch.Count; i++)
            Array.Copy(batch[i].expectedOutputs, 0, _flattenedExpectedOutputs, i * outputLayer.LayerData.NumOutputs,
                outputLayer.LayerData.NumOutputs);

        _buffers.Errors.View.MemSetToZero();
        _buffers.Outputs.View.CopyFromCPU(_flattenedExpectedOutputs);
        _lossFunctionKernel(_buffers.BatchSize * outputLayer.LayerData.NumOutputs, _network.NetworkData,
            outputLayer.LayerData, _buffers.Outputs.View, _buffers.Activations.View, _buffers.Errors.View);
        _buffers.Outputs.View.CopyToCPU(_flattenedExpectedOutputs);
        var sampleError = _flattenedExpectedOutputs.Sum();

        // Backward Pass
        for (var i = _network._layers.Length - 1; i >= 0; i--)
        {
            _network._layers[i].Backward(this);
            _accelerator.Synchronize();
        }

        _optimizer.Optimize(_network.NetworkData, _buffers, learningRate);
        _accelerator.Synchronize();

        //Console.WriteLine($" final sync: {stopwatch.ElapsedMilliseconds}ms");
        stopwatch.Restart();
        return sampleError / finalLayer.NumOutputs;
    }
    
    
    public void TrainAccelerated(List<(float[] imageData, float[] label)> data, TrainConfig trainConfig)
    {
        Console.WriteLine("Training network");
        Reset();

        var learningRateScheduler = LearningRateSchedulerFactory.Create(trainConfig.Scheduler);
        var batchSize = _buffers.BatchSize;
        var totalBatches = (int)Math.Ceiling((double)data.Count / batchSize);

        var stopwatch = Stopwatch.StartNew();
        for (var epoch = 0; epoch < trainConfig.Epochs; epoch++)
        {
            // Shuffle data in-place with Fisher-Yates for efficiency
            for (int i = data.Count - 1; i > 0; i--)
            {
                int j = Random.Shared.Next(i + 1);
                (data[i], data[j]) = (data[j], data[i]);
            }

            float epochError = 0;
            int sampleCount = 0;
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

            Console.WriteLine($"Epoch {epoch}, LR: {learningRate.ToSignificantFigures(3)} Average MSE: {averageMSE.ToSignificantFigures(3)}, Time: {elapsedTime}ms, {samplesPerSec}/s");
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
        
        // Precision, Recall, F1-Score for each label
        for (int i = 0; i < truePositives.Length; i++)
        {
            float precision = Precision(truePositives[i], falsePositives[i]);
            float recall = Recall(truePositives[i], falseNegatives[i]);
            float f1Score = F1Score(precision, recall);

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