using vcortex.cpu;
using vcortex.gpu;
using vcortex.Input;
using vcortex.Layers;
using vcortex.Network;
using vcortex.Optimizers;
using vcortex.Training;

namespace vcortex.console;

internal class Program
{
    private static void Main(string[] args)
    {
        Console.WriteLine("vcortex");
        var inputConfig = new ConvolutionInputConfig
        {
            Width = 28,
            Height = 28,
            Grayscale = true
        };
        var (trainData, testData) = DataLoader.LoadMNIST(inputConfig, "../../../../data/mnist_fashion/train.csv", "../../../../data/mnist_fashion/test.csv", 10);

        var network = new NetworkBuilder(inputConfig)
            .Add(new Convolution
            {
                Activation = ActivationType.LeakyRelu,
                KernelSize = 3,
                KernelsPerChannel = 32,
                Padding = 1,
                Stride = 1
            })
            .Add(new Maxpool
            {
                PoolSize = 2
            })
            .Add(new Dense
            {
                Activation = ActivationType.LeakyRelu,
                Neurons = 128
            })
            .Add(new Dropout
            {
                DropoutRate = 0.2f
            })
            .Add(new Softmax
            {
                Neurons = 10
            })
            .Build();
        
        Console.WriteLine($"parameters: {network.NetworkData.ParameterCount}");
        Console.WriteLine($"activations: {network.NetworkData.ActivationCount}");
        
        var trainingConfig = new TrainConfig
        {
            Epochs = 20,
            Scheduler = new ExponentialDecay()
            {
                InitialLearningRate = 0.001f,
                DecayRate = 0.05f
            },
            Optimizer = new Adam(),
            LossFunction = LossFunction.Mse,
            BatchSize = 100
        };
        
        //var accelerator = new CpuNetworkTrainer(network, trainingConfig);
        var accelerator = new GpuNetworkTrainer(GpuType.Cuda, 0, network, trainingConfig);
        //accelerator.InitRandomParameters();
        accelerator.ReadParametersFromDisk("../../../../data/weights.bin");

        accelerator.Train(trainData);
        accelerator.Test(testData, 0.1f);
        
        accelerator.SaveParametersToDisk("../../../../data/weights.bin");
    }
}