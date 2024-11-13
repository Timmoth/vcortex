using vcortex.gpu;
using vcortex.Input;
using vcortex.Layers;
using vcortex.LearningRate;
using vcortex.Network;
using vcortex.Optimizers;
using vcortex.Training;

namespace vcortex.console;

internal class Program
{
    private static readonly List<TrainConfig> TrainConfigs = new()
    {
        new TrainConfig
        {
            TrainPath = "../../../../data/mnist_digits/train.csv",
            TestPath = "../../../../data/mnist_digits/test.csv",
            Outputs = 10,
            Epochs = 10,
            Scheduler = new ConstantLearningRate
            {
                LearningRate = 0.001f
            },
            Optimizer = new Adam(),
            InputDateType = InputDateType.Csv,
            InputConfig = new ConvolutionInputConfig
            {
                Width = 28,
                Height = 28,
                Grayscale = true
            }
        },
        new TrainConfig
        {
            TrainPath = "../../../../data/mnist_fashion/train.csv",
            TestPath = "../../../../data/mnist_fashion/test.csv",
            Outputs = 10,
            Epochs = 10,
            Scheduler = new ExponentialDecay
            {
                InitialLearningRate = 0.002f,
                DecayRate = 0.05f
            },
            Optimizer = new Adam(),
            InputDateType = InputDateType.Csv,
            InputConfig = new ConvolutionInputConfig
            {
                Width = 28,
                Height = 28,
                Grayscale = true
            }
        },
        new TrainConfig
        {
            TrainPath = "../../../../data/mnist_sign/train.csv",
            TestPath = "../../../../data/mnist_sign/test.csv",
            Outputs = 25,
            Epochs = 10,
            Scheduler = new ConstantLearningRate
            {
                LearningRate = 0.001f
            },
            Optimizer = new Adam(),
            InputDateType = InputDateType.Csv,
            InputConfig = new ConvolutionInputConfig
            {
                Width = 28,
                Height = 28,
                Grayscale = true
            }
        },
        new TrainConfig
        {
            TrainPath = "../../../../data/pandas_or_bears/train",
            TestPath = "../../../../data/pandas_or_bears/test",
            Outputs = 2,
            Epochs = 10,
            Scheduler = new ConstantLearningRate
            {
                LearningRate = 0.001f
            },
            Optimizer = new Adam(),
            InputDateType = InputDateType.Directory,
            InputConfig = new ConvolutionInputConfig
            {
                Width = 64,
                Height = 64,
                Grayscale = false
            }
        }
    };

    private static void Main(string[] args)
    {
        Console.WriteLine("vcortex");
        var trainConfig = TrainConfigs[3];

        var net = new NetworkBuilder(trainConfig.InputConfig)
            .Add(new Convolution
            {
                Activation = ActivationType.LeakyRelu,
                KernelSize = 3,
                KernelsPerChannel = 16,
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
                Neurons = trainConfig.Outputs
            })
            .Build();

        // var convolutionConfig = trainConfig.InputConfig as ConvolutionInputConfig;
        // var net = new NetworkBuilder(new ConnectedInputConfig()
        // {
        //     NumInputs = convolutionConfig.Width * convolutionConfig.Height
        // })
        //     .Add(new DenseLayer(512, ActivationType.Sigmoid))
        //     //.Add(new DropoutLayer(0.1f))
        //     .Add(new SoftmaxConnectedLayer(trainConfig.Outputs))
        //     .Build();

        Console.WriteLine($"parameters: {net.NetworkData.ParameterCount}");
        Console.WriteLine($"activations: {net.NetworkData.ActivationCount}");

        var (train, test) = trainConfig.InputDateType == InputDateType.Csv
            ? DataLoader.LoadMNIST(trainConfig)
            : DataLoader.LoadData(trainConfig);

        var accelerator = new NetworkTrainer(net, LossFunction.CrossEntropyLoss, trainConfig.Optimizer, 100);
        //accelerator.InitRandomWeights();
        accelerator.ReadFromDisk("../../../../data/weights.bin");

        accelerator.TrainAccelerated(train, trainConfig);

        accelerator.SaveToDisk("../../../../data/weights.bin");

        accelerator.Test(test, 0.1f);
    }
}