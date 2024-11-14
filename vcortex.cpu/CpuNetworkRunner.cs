using vcortex.cpu.Layers;
using vcortex.Layers;
using vcortex.Network;

namespace vcortex.cpu;

public class CpuNetworkRunner : ICpuNetworkAgent
{
    private readonly float[] _flattenedInputs;
    private readonly ILayer[] _layers;

    public CpuNetworkRunner(NetworkConfig network, int batchSize)
    {
        Network = network;

        Buffers = new NetworkAcceleratorBuffers(network, batchSize);
        _layers = network.Layers.Select(l => CpuLayerFactory.Create(l, Buffers, network.NetworkData)).ToArray();

        var inputLayer = _layers[0];
        var inputCount = inputLayer.Config.NumInputs * Buffers.BatchSize;
        _flattenedInputs = new float[inputCount];
    }

    public NetworkConfig Network { get; }

    public NetworkAcceleratorBuffers Buffers { get; }

    public bool IsTraining => false;


    public void Dispose()
    {
        Buffers.Dispose();
    }

    public List<float[]> Predict(List<float[]> batchs)
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

            foreach (var layer in _layers)
            {
                layer.Forward();
            }

            for (var i = 0; i < batch.Count; i++)
            {
                var output = new float[finalLayer.Config.NumOutputs];
                outputs.Add(output);
            }
        }

        return outputs;
    }
    
    #region Io

    public void SaveParametersToDisk(string filePath)
    {
        using var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write);
        using var writer = new BinaryWriter(stream);
        // Write the number of arrays to allow easy deserialization
        writer.Write(Network.NetworkData.ParameterCount);
        foreach (var value in Buffers.Parameters) writer.Write(value);
    }

    public void ReadParametersFromDisk(string filePath)
    {
        using var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
        using var reader = new BinaryReader(stream);
        // Read the number of arrays
        var length = reader.ReadInt32();

        for (var j = 0; j < length; j++) Buffers.Parameters[j] = reader.ReadSingle();
    }

    public float[] GetParameters()
    {
        var parameters = new float[Network.NetworkData.ParameterCount];
        Array.Copy(Buffers.Parameters, 0, parameters, 0, parameters.Length);
        return parameters;
    }

    public void LoadParameters(float[] parameters)
    {
        Array.Copy(parameters, 0, Buffers.Parameters,0, parameters.Length);
    }

    #endregion
}