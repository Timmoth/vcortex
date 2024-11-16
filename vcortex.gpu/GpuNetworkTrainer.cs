using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using vcortex.gpu.Layers;
using vcortex.gpu.LossFunctions;
using vcortex.gpu.Optimizers;
using vcortex.Layers;
using vcortex.Network;
using vcortex.Optimizers;
using vcortex.Training;

namespace vcortex.gpu;

public class GpuNetworkTrainer : NetworkTrainerBase
{
    #region Props

    private readonly Context _context;
    private readonly float[] _flattenedExpectedOutputs;
    private readonly float[] _flattenedInputs;
    private readonly ILayer[] _layers;
    private readonly Action<Index1D, NetworkData, ArrayView<float>, ArrayView<float>, int> _loadInputsKernel;
    private readonly Action<Index1D, NetworkData, ArrayView<float>, ArrayView<float>, int, int>
        _outputKernel;
    private readonly IOptimizer _optimizer;
    private readonly TrainConfig _trainingConfig;
    public Accelerator Accelerator { get; }
    public NetworkConfig Network { get; }
    public NetworkBuffers Buffers { get; }
    private readonly ILossFunction _lossFunction;
    protected override ILayer[] Layers => _layers;
    protected override TrainConfig TrainingConfig => _trainingConfig;
    protected override IOptimizer Optimizer => _optimizer;
    protected override ILossFunction LossFunction => _lossFunction;
    #endregion
    
    public GpuNetworkTrainer(GpuType gpuType, int gpuIndex, NetworkConfig network, TrainConfig trainingConfig)
    {
        Network = network;
        _trainingConfig = trainingConfig;
        _context = Context.Create(b => { b.Default().EnableAlgorithms().Math(MathMode.Fast); });

        if (gpuType == GpuType.Cuda)
        { 
            Accelerator = _context.CreateCudaAccelerator(gpuIndex);
        }
        else
        {
            Accelerator = _context.CreateCLAccelerator(gpuIndex);
        }

        Buffers = new NetworkBuffers(Accelerator, network, trainingConfig.BatchSize);

        _layers = network.Layers.Select(l => GpuLayerFactory.Create(Buffers, Accelerator, network.NetworkData, l)).ToArray();
        _optimizer = GpuOptimizerFactory.Create(trainingConfig.Optimizer, this);

        _loadInputsKernel =
            Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, ArrayView<float>, ArrayView<float>, int>(
                    LoadInputs);        
        
        _outputKernel =
            Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, NetworkData, ArrayView<float>, ArrayView<float>, int, int>(
                    OutputKernel);

        if (trainingConfig.LossFunction == vcortex.Training.LossFunction.Mse)
        {
            _lossFunction = new GpuMseLoss(this);
        }
        else
        {
            _lossFunction = new GpuCrossEntropyLoss(this);
        }

        var inputLayer = _layers[0];
        var inputCount = inputLayer.Config.NumInputs * Buffers.BatchSize;
        _flattenedInputs = new float[inputCount];

        var outputLayer = _layers[^1];
        var outputCount = outputLayer.Config.NumOutputs * Buffers.BatchSize;
        _flattenedExpectedOutputs = new float[outputCount];
        
        Console.WriteLine($"Device: '{Accelerator.Device.Name}'");
    }
    public override void Dispose()
    {
        Accelerator.Dispose();
        _context.Dispose();
        Buffers.Dispose();
        _optimizer.Dispose();
        _lossFunction.Dispose();
    }

    protected override List<float[]> Predict(List<float[]> batches)
    {
        var outputs = new List<float[]>();
        var batchSize = Buffers.BatchSize;
        var finalLayer = _layers[^1];

        // Divide the data into mini-batches
        for (var batchStart = 0; batchStart < batches.Count; batchStart += batchSize)
        {
            // Get the current batch
            var batch = batches.Skip(batchStart).Take(batchSize).ToList();
            var inputLayer = _layers[0];
            for (var i = 0; i < batch.Count; i++)
                Array.Copy(batch[i], 0, _flattenedInputs, i * inputLayer.Config.NumInputs,
                    inputLayer.Config.NumInputs);

            Buffers.Inputs.View.CopyFromCPU(_flattenedInputs);
            _loadInputsKernel(_flattenedInputs.Length, Network.NetworkData, Buffers.Inputs.View,
                Buffers.Activations.View, inputLayer.Config.NumInputs);

            foreach (var layer in _layers)
            {
                layer.Forward();
                Accelerator.Synchronize();
            }

            _outputKernel(Buffers.BatchSize * finalLayer.Config.NumOutputs, Network.NetworkData,
                Buffers.Outputs.View, Buffers.Activations.View, finalLayer.Config.NumOutputs,
                finalLayer.Config.ActivationOutputOffset);
            Buffers.Outputs.View.CopyToCPU(_flattenedExpectedOutputs);
            
            for (var i = 0; i < batch.Count; i++)
            {
                var output = new float[finalLayer.Config.NumOutputs];
                Array.Copy(_flattenedExpectedOutputs, i * finalLayer.Config.NumOutputs, output, 0,
                    finalLayer.Config.NumOutputs);
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
    
    public static void OutputKernel(
        Index1D index,
        NetworkData networkData,
        ArrayView<float> outputs,
        ArrayView<float> activations,
        int numOutputs, int activationOutputOffset)
    {
        // Number of samples in the batch
        var batchIndex = index / numOutputs;
        var outputIndex = index % numOutputs;
        var activationOutputIndex = batchIndex * networkData.ActivationCount + activationOutputOffset + outputIndex;
        
        outputs[numOutputs * batchIndex + outputIndex] = activations[activationOutputIndex];
    }

    protected override void Reset()
    {
        _optimizer.Reset();
    }
    
    protected override void InitBatch(List<(float[] inputs, float[] expectedOutputs)> batch)
    {        
        var inputLayer = _layers[0];

        for (var i = 0; i < batch.Count; i++)
            Array.Copy(batch[i].inputs, 0, _flattenedInputs, i * inputLayer.Config.NumInputs,
                inputLayer.Config.NumInputs);

        Buffers.Inputs.View.CopyFromCPU(_flattenedInputs);
        _loadInputsKernel(_flattenedInputs.Length, Network.NetworkData, Buffers.Inputs.View,
            Buffers.Activations.View, inputLayer.Config.NumInputs);
    }
    
    #region Io

    public override void SaveParametersToDisk(string filePath)
    {
        using var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write);
        using var writer = new BinaryWriter(stream);
        // Write the number of arrays to allow easy deserialization
        writer.Write(Network.NetworkData.ParameterCount);
        var parameters = new float[Network.NetworkData.ParameterCount];
        Buffers.Parameters.View.CopyToCPU(parameters);
        foreach (var value in parameters) writer.Write(value);
        
    }

    public override void ReadParametersFromDisk(string filePath)
    {
        using var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
        using var reader = new BinaryReader(stream);
        // Read the number of arrays
        var length = reader.ReadInt32();
        var parameters = new float[length];

        for (var j = 0; j < length; j++) parameters[j] = reader.ReadSingle();
        Buffers.Parameters.View.CopyFromCPU(parameters);
    }

    public override float[] GetParameters()
    {
        var parameters = new float[Network.NetworkData.ParameterCount];
        Buffers.Parameters.View.CopyToCPU(parameters);
        return parameters;
    }

    public override void LoadParameters(float[] parameters)
    {
        Buffers.Parameters.View.CopyFromCPU(parameters);
    }

    #endregion
}