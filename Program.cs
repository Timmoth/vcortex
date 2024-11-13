using vcortex.Accelerated;
using vcortex.Layers.Connected;
using vcortex.Layers.Convolution;

namespace vcortex;

internal class Program
{
    private static readonly List<TrainConfig> TrainConfigs = new()
    {
        new TrainConfig
        {
            TrainPath = "../../../mnist_digits/train.csv",
            TestPath = "../../../mnist_digits/test.csv",
            Outputs = 10,
            Epochs = 10,
            LearningRate = 0.001f,
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
            TrainPath = "../../../mnist_fashion/train.csv",
            TestPath = "../../../mnist_fashion/test.csv",
            Outputs = 10,
            Epochs = 20,
            LearningRate = 0.001f,
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
            TrainPath = "../../../mnist_sign/train.csv",
            TestPath = "../../../mnist_sign/test.csv",
            Outputs = 25,
            Epochs = 10,
            LearningRate = 0.001f,
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
            TrainPath = "../../../pandas_or_bears/train",
            TestPath = "../../../pandas_or_bears/test",
            Outputs = 2,
            Epochs = 80,
            LearningRate = 0.001f,
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
        var trainConfig = TrainConfigs[1];

        var net = new NetworkBuilder(trainConfig.InputConfig)
            .Add(new KernelConvolutionLayer(1, 0, 32))
            //.Add(new ReLUConvolutionLayer())
            .Add(new MaxPoolingConvolutionLayer(2))

            //.Add(new DropoutConnectedLayer(0.1f))
            .Add(new SigmoidConnectedLayer(64))
            .Add(new DropoutConnectedLayer(0.2f))
            .Add(new SoftmaxConnectedLayer(trainConfig.Outputs))
            .Build(100);

        //var convolutionConfig = trainConfig.InputConfig as ConvolutionInputConfig;
        //var net = new NetworkBuilder(new ConnectedInputConfig()
        //{
        //    NumInputs = convolutionConfig.Width * convolutionConfig.Height
        //})
        //    .Add(new SigmoidConnectedLayer(512))
        //    .Add(new DropoutConnectedLayer(0.1f))
        //    .Add(new SoftmaxConnectedLayer(trainConfig.Outputs))
        //    .Build(100);

        Console.WriteLine($"parameters: {net.ParameterCount}");
        Console.WriteLine($"activations: {net.ActivationCount}");

        var (train, test) = trainConfig.InputDateType == InputDateType.Csv
            ? DataLoader.LoadMNIST(trainConfig)
            : DataLoader.LoadData(trainConfig);

        var accelerator = new NetworkAccelerator(net);
        accelerator.InitWeights();

        Trainer.TrainAccelerated(accelerator, train, trainConfig.Epochs);

        //Trainer.TrainBatched(net, train, trainConfig.Epochs, trainConfig.LearningRate, 16);
        Trainer.Test(accelerator, test, 0.1f);
    }
}