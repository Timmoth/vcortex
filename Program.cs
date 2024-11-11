using vcortex.Accelerated;
using vcortex.Layers.Connected;
using vcortex.Layers.Convolution;

namespace vcortex;

internal class Program
{
    private static readonly List<TrainConfig> TrainConfigs = new()
    {
        new()
        {
            TrainPath = "../../../mnist_digits/train.csv",
            TestPath = "../../../mnist_digits/test.csv",
            Outputs = 10,
            Epochs = 10,
            LearningRate = 0.02f,
            InputDateType = InputDateType.Csv,
            InputConfig = new ConvolutionInputConfig
            {
                Width = 28,
                Height = 28,
                Grayscale = true
            }
        },
        new()
        {
            TrainPath = "../../../mnist_fashion/train.csv",
            TestPath = "../../../mnist_fashion/test.csv",
            Outputs = 10,
            Epochs = 20,
            LearningRate = 0.02f,
            InputDateType = InputDateType.Csv,
            InputConfig = new ConvolutionInputConfig
            {
                Width = 28,
                Height = 28,
                Grayscale = true
            }
        },
        new()
        {
            TrainPath = "../../../pandas_or_bears/train",
            TestPath = "../../../pandas_or_bears/test",
            Outputs = 2,
            Epochs = 20,
            LearningRate = 0.001f,
            InputDateType = InputDateType.Directory,
            InputConfig = new ConvolutionInputConfig
            {
                Width = 32,
                Height = 32,
                Grayscale = true
            }
        }
    };

    private static void Main(string[] args)
    {
        Console.WriteLine("vcortex");
        var trainConfig = TrainConfigs[0];

        //var net = new NetworkBuilder(trainConfig.InputConfig)
        //    .Add(new KernelConvolutionLayer(32))
        //    .Add(new LeakyReLUConvolutionLayer())
        //    .Add(new MaxPoolingConvolutionLayer(2))
        //    .Add(new LeakyReluConnectedLayer(32))
        //    .Add(new LeakyReluConnectedLayer(32))
        //    .Add(new SoftmaxConnectedLayer(trainConfig.Outputs))
        //    .Build();

        var net = new NetworkBuilder(trainConfig.InputConfig)
            .Add(new KernelConvolutionLayer(12))
            .Add(new ReLUConvolutionLayer())
            .Add(new MaxPoolingConvolutionLayer(2))
            .Add(new SigmoidConnectedLayer(128))
            .Add(new SoftmaxConnectedLayer(trainConfig.Outputs))
            .Build();


        //var net = new NetworkBuilder(new ConnectedInputConfig()
        //{
        //    NumInputs = 28 * 28
        //})
        //    .Add(new SigmoidConnectedLayer(64))
        //    .Add(new SoftmaxConnectedLayer(trainConfig.Outputs))
        //    .Build();

        Console.WriteLine($"parameters: {net.ParameterCount}");
        Console.WriteLine($"activations: {net.ActivationCount}");

        var (train, test) = trainConfig.InputDateType == InputDateType.Csv
            ? DataLoader.LoadMNIST(trainConfig)
            : DataLoader.LoadData(trainConfig);

        var accelerator = new NetworkAccelerator(net);
        accelerator.InitWeights();

        Trainer.TrainAccelerated(accelerator, train, 15, 0.2f);

        //Trainer.TrainBatched(net, train, trainConfig.Epochs, trainConfig.LearningRate, 16);
        Trainer.Test(accelerator, test, 0.1f);
    }
}