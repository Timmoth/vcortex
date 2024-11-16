using vcortex.Layers;
using vcortex.Network;

namespace vcortex.cpu.Layers;

public static class CpuLayerFactory
{
    public static ILayer Create(Layer layer, NetworkBuffers buffers, NetworkData networkData)
    {
        switch (layer)
        {
            case Dense config:
                return new DenseLayer(config, buffers, networkData);

            case Convolution config:
                return new KernelConvolutionLayer(config, buffers, networkData);

            case Dropout config:
                return new DropoutLayer(config, buffers, networkData);

            case Maxpool config:
                return new MaxPoolLayer(config, buffers, networkData);

            case Softmax config:
                return new SoftmaxConnectedLayer(config, buffers, networkData);

            default:
                throw new ArgumentException($"Unsupported optimizer type: {layer.GetType().Name}");
        }
    }
}