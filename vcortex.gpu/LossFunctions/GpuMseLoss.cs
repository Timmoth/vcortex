using ILGPU;
using ILGPU.Runtime;
using vcortex.Network;

namespace vcortex.gpu.LossFunctions;

public class GpuMseLoss : ILossFunction
{
    private readonly GpuNetworkTrainer _trainer;
    private readonly float[] _flattenedExpectedOutputs;

    private readonly Action<Index1D, NetworkData, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>
        _lossFunctionKernel;
    
    public GpuMseLoss(GpuNetworkTrainer trainer)
    {
        _trainer = trainer;
        _lossFunctionKernel =
            trainer.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, ArrayView<float>, ArrayView<float>,
                    ArrayView<float>, int, int, int>(
                    MSE);
        
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
}