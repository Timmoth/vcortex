using ILGPU;
using ILGPU.Runtime;
using vcortex.Network;

namespace vcortex.gpu.LossFunctions;

public class GpuCrossEntropyLoss : ILossFunction
{
    private readonly GpuNetworkTrainer _trainer;
    private readonly float[] _flattenedExpectedOutputs;

    private readonly Action<Index1D, NetworkData, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>
        _lossFunctionKernel;
    
    public GpuCrossEntropyLoss(GpuNetworkTrainer trainer)
    {
        _trainer = trainer;
        _lossFunctionKernel =
            trainer.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, int, int, int>(
                    CrossEntropyLoss);
        
        var outputLayer = trainer.Network.Layers[^1];
        var outputCount = outputLayer.NumOutputs * trainer.Buffers.BatchSize;
        _flattenedExpectedOutputs = new float[outputCount];
    }
    public void Dispose()
    {
        
    }

    public float Apply(List<(float[] inputs, float[] expectedOutputs)> batch)
    {
        var outputLayer = _trainer.Network.Layers[^1];
        
        for (var i = 0; i < batch.Count; i++)
            Array.Copy(batch[i].expectedOutputs, 0, _flattenedExpectedOutputs, i * outputLayer.NumOutputs,
                outputLayer.NumOutputs);

        _trainer.Buffers.Errors.View.MemSetToZero();
        _trainer.Buffers.Outputs.View.CopyFromCPU(_flattenedExpectedOutputs);
        _lossFunctionKernel(_trainer.Buffers.BatchSize * outputLayer.NumOutputs, _trainer.Network.NetworkData,
            _trainer. Buffers.Outputs.View, _trainer.Buffers.Activations.View, _trainer.Buffers.Errors.View, outputLayer.NumOutputs,
            outputLayer.ActivationOutputOffset, outputLayer.NextLayerErrorOffset);
        _trainer.Buffers.Outputs.View.CopyToCPU(_flattenedExpectedOutputs);
        
        return _flattenedExpectedOutputs.Sum() / outputLayer.NumOutputs;
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

        // Store the loss in the outputs array
        outputs[numOutputs * batchIndex + outputIndex] = loss;

        // Calculate the gradient of the loss w.r.t. the predicted probability (backpropagation)
        // Derivative of cross-entropy loss with softmax is: p - y
        var gradient = actual - expected;

        // Store the gradient in the errors array
        errors[nextErrorOffset + outputIndex] = gradient;
    }
}