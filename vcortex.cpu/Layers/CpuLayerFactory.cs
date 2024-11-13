using vcortex.Layers;

namespace vcortex.cpu.Layers;

public static class CpuLayerFactory
{
    public static ILayer Create(Layer layer)
    {
        switch (layer)
        {
            case Dense config:
                return new DenseLayer(config);

            case Convolution config:
                return new KernelConvolutionLayer(config);

            case Dropout config:
                return new DropoutLayer(config);

            case Maxpool config:
                return new MaxPoolLayer(config);

            case Softmax config:
                return new SoftmaxConnectedLayer(config);

            default:
                throw new ArgumentException($"Unsupported optimizer type: {layer.GetType().Name}");
        }
    }
}