using ILGPU.Runtime;
using vcortex.Layers;
using vcortex.Network;

namespace vcortex.gpu.Layers;

public static class GpuLayerFactory
{
    public static ILayer Create(NetworkAcceleratorBuffers buffers, Accelerator accelerator, NetworkData networkData, Layer layer)
    {
        switch (layer)
        {
            case Dense config:
                return new DenseLayer(config, buffers, accelerator, networkData);

            case Convolution config:
                return new KernelConvolutionLayer(config, buffers, accelerator, networkData);

            case Dropout config:
                return new DropoutLayer(config, buffers, accelerator, networkData);

            case Maxpool config:
                return new MaxPoolLayer(config, buffers, accelerator, networkData);

            case Softmax config:
                return new SoftmaxConnectedLayer(config, buffers, accelerator, networkData);

            default:
                throw new ArgumentException($"Unsupported optimizer type: {layer.GetType().Name}");
        }
    }
}