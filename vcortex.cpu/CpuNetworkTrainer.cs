using vcortex.cpu.Layers;
using vcortex.cpu.LossFunctions;
using vcortex.cpu.Optimizers;
using vcortex.Layers;
using vcortex.Network;
using vcortex.Optimizers;
using vcortex.Training;

namespace vcortex.cpu;

public class CpuNetworkTrainer : NetworkTrainerBase
{
    #region Props

    private readonly ILayer[] _layers;
    private readonly IOptimizer _optimizer;
    public NetworkConfig Network { get; }
    public NetworkBuffers Buffers { get; }
    private readonly ILossFunction _lossFunction;
    private readonly TrainConfig _trainConfig;
    protected override ILayer[] Layers => _layers;
    protected override TrainConfig TrainingConfig  => _trainConfig;
    protected override IOptimizer Optimizer => _optimizer;
    protected override ILossFunction LossFunction => _lossFunction;
    #endregion

    
    public CpuNetworkTrainer(NetworkConfig network, TrainConfig trainingConfig)
    {
        Network = network;
        _trainConfig = trainingConfig;
        Buffers = new NetworkBuffers(network, trainingConfig.BatchSize);
        _optimizer = CpuOptimizerFactory.Create(trainingConfig.Optimizer, Buffers, network.NetworkData);
        _layers = network.Layers.Select(l => CpuLayerFactory.Create(l, Buffers, network.NetworkData)).ToArray();

        if (trainingConfig.LossFunction == vcortex.Training.LossFunction.Mse)
        {
            _lossFunction = new CpuMseLoss(Buffers, network);
        }
        else
        {
            _lossFunction = new CpuCrossEntropyLoss(Buffers, network);
        }
        
        Console.WriteLine($"Device: 'cpu'");
    }
    public override void Dispose()
    {
        Buffers.Dispose();
        _lossFunction.Dispose();
        _optimizer.Dispose();
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
                Array.Copy(batch[i], 0, Buffers.Activations, i * Network.NetworkData.ActivationCount,
                    inputLayer.Config.NumInputs);

            foreach (var layer in _layers)
            {
                layer.Forward();
            }

            for (var i = 0; i < batch.Count; i++)
            {
                var output = new float[finalLayer.Config.NumOutputs];
                Array.Copy(Buffers.Activations, i * Network.NetworkData.ActivationCount + finalLayer.Config.ActivationOutputOffset, output, 0,
                    finalLayer.Config.NumOutputs);
                outputs.Add(output);
            }
        }

        return outputs;
    }


    protected override void Reset()
    {
        _optimizer.Reset();
    }

    protected override void InitBatch(List<(float[] inputs, float[] expectedOutputs)> batch)
    {        
        var inputLayer = _layers[0];
        for (var i = 0; i < batch.Count; i++)
            Array.Copy(batch[i].inputs, 0, Buffers.Activations, i * Network.NetworkData.ActivationCount,
                inputLayer.Config.NumInputs);
    }
    
    #region Io

    public override void SaveParametersToDisk(string filePath)
    {
        using var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write);
        using var writer = new BinaryWriter(stream);
        // Write the number of arrays to allow easy deserialization
        writer.Write(Network.NetworkData.ParameterCount);
        foreach (var value in Buffers.Parameters) writer.Write(value);
    }

    public override void ReadParametersFromDisk(string filePath)
    {
        using var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
        using var reader = new BinaryReader(stream);
        // Read the number of arrays
        var length = reader.ReadInt32();

        for (var j = 0; j < length; j++) Buffers.Parameters[j] = reader.ReadSingle();
    }

    public override float[] GetParameters()
    {
        var parameters = new float[Network.NetworkData.ParameterCount];
        Array.Copy(Buffers.Parameters, 0, parameters, 0, parameters.Length);
        return parameters;
    }

    public override void LoadParameters(float[] parameters)
    {
        Array.Copy(parameters, 0, Buffers.Parameters,0, parameters.Length);
    }

    #endregion
}