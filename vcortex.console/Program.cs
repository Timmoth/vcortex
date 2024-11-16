using System.Text.Json;
using vcortex.cpu;
using vcortex.gpu;
using vcortex.Input;
using vcortex.Layers;
using vcortex.Network;
using vcortex.Training;

namespace vcortex.console;

internal class Program
{
    private static void Main(string[] args)
    {
        // var testConfig = new Config()
        // {
        //     Network = new NetworkBuilder( new ConvolutionInputConfig
        //         {
        //             Width = 28,
        //             Height = 28,
        //             Grayscale = true
        //         })
        //         .Add(new Convolution
        //         {
        //             Activation = ActivationType.LeakyRelu,
        //             KernelSize = 3,
        //             KernelsPerChannel = 32,
        //             Padding = 1,
        //             Stride = 1
        //         })
        //         .Add(new Maxpool
        //         {
        //             PoolSize = 2
        //         })
        //         .Add(new Dense
        //         {
        //             Activation = ActivationType.LeakyRelu,
        //             Neurons = 128
        //         })
        //         .Add(new Dropout
        //         {
        //             DropoutRate = 0.2f
        //         })
        //         .Add(new Softmax
        //         {
        //             Neurons = 10
        //         })
        //         .Build(),
        //     Training = new TrainConfig
        //     {
        //         Epochs = 20,
        //         Scheduler = new ExponentialDecay()
        //         {
        //             InitialLearningRate = 0.001f,
        //             DecayRate = 0.05f
        //         },
        //         Optimizer = new Adam(),
        //         LossFunction = LossFunction.Mse,
        //         BatchSize = 100
        //     }
        // };
        // var jsonContent1 =  JsonSerializer.Serialize(testConfig, NetworkConfig.SerializerOptions) ?? throw new Exception("Failed to deserialize network config");
        // File.WriteAllText("network.json", jsonContent1);

        Console.WriteLine("vcortex");
        
        var jsonContent = File.ReadAllText("network.json");
        var config =  JsonSerializer.Deserialize<Config>(jsonContent, Utils.SerializerOptions) ?? throw new Exception("Failed to deserialize network config");
     
        var trainData = DataLoader.LoadCsv(config.TrainingFile, config.Network.Input as ConvolutionInputConfig, config.Network.OutputCount);
        
        Console.WriteLine($"parameters: {config.Network.NetworkData.ParameterCount}");
        Console.WriteLine($"activations: {config.Network.NetworkData.ActivationCount}");
        
        INetworkTrainerAgent trainerAgent;
        
        if (config.Training.Platform == Platform.Cuda)
        {
            trainerAgent = new GpuNetworkTrainer(GpuType.Cuda, config.Training.GpuIndex ?? 0, config.Network, config.Training);
        }else if(config.Training.Platform == Platform.OpenCl)
        {
            trainerAgent = new GpuNetworkTrainer(GpuType.OpenCl, config.Training.GpuIndex ?? 0, config.Network, config.Training);
        }
        else if(config.Training.Platform == Platform.Cpu)
        {
            trainerAgent = new CpuNetworkTrainer(config.Network, config.Training);
        }
        else
        {
            Console.WriteLine("Invalid platform");
            return;
        }
        
        if (File.Exists(config.ParametersFile))
        {
            trainerAgent.ReadParametersFromDisk(config.ParametersFile);
        }
        else
        {
            trainerAgent.InitRandomParameters();
        }

        if (File.Exists(config.TrainingFile))
        {
            trainerAgent.Train(trainData);
        }

        if (File.Exists(config.TestingFile))
        {
            var testData = DataLoader.LoadCsv(config.TestingFile, config.Network.Input as ConvolutionInputConfig,config.Network.OutputCount);
            trainerAgent.Test(testData, 0.1f);
        }
        
        trainerAgent.SaveParametersToDisk(config.ParametersFile);
    }
}