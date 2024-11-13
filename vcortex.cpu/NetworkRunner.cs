using vcortex.cpu.Layers;
using vcortex.Network;

namespace vcortex.cpu;

public class NetworkRunner : INetworkAgent
{
    private readonly float[] _flattenedInputs;
    private readonly ILayer[] _layers;

    public NetworkRunner(NetworkConfig network, int batchSize)
    {
        Network = network;
        _layers = network.Layers.Select(CpuLayerFactory.Create).ToArray();

        Buffers = new NetworkAcceleratorBuffers(network, batchSize);

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
                layer.Forward(this);
            }

            for (var i = 0; i < batch.Count; i++)
            {
                var output = new float[finalLayer.Config.NumOutputs];
                outputs.Add(output);
            }
        }

        return outputs;
    }
}