# vcortex
Lightweight CPU / GPU machine learning for dotnet

## Image Classification Quickstart
```csharp
// Define the input structure
var inputConfig = new ConvolutionInputConfig
{
    Width = 28,
    Height = 28,
    Grayscale = true
};

// Define your networks architecture
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

// Define the training parameters   
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

// Create a CPU trainer
var trainer = new CpuNetworkTrainer(network, trainingConfig);
// Or create a GPU trainer
var trainer = new GpuNetworkTrainer(GpuType.Cuda, 0, net, trainingConfig);

// Initialize the trainable parameters to random values
trainer.InitRandomParameters();
// Train
trainer.Train(trainData);
// Test
trainer.Test(testData, 0.1f);
```

## Persistence
```csharp

// Load the network architecture from disk
var network = NetworkConfig.DeserializeFromDisk("./network.json");
// Save the network architecture to disk
network.SerializeToDisk("./network.json");

// Load the networks parameters from disk
trainer.SaveParametersToDisk("./weights.bin");
// Save the networks parameters to disk
trainer.ReadParametersFromDisk("./weights.bin");

// Load the networks parameters into an array
float[] parameters = trainer.GetParameters();
// Load the networks parameters from an array
trainer.LoadParameters(parameters);
```

## Layers

### Convolution
```json
{
    "$type": "convolution",     // The layer type
    "stride": 1,                // How many pixels to move the kernel accross each step
    "padding": 1,               // How many pixels to pad the edges of the image
    "kernels_per_channel": 32,  // How many kernels to apply to each channel
    "kernel_size": 3,           // The pixel width & height of the kernel
    "activation": 2             // The activation function used
}
```

### Dense
```json
{
    "$type": "dense",
    "activation": 2,
    "neurons": 128
}
```

### Dropout
```json
{
    "$type": "dropout",
    "dropout_rate": 0.2
}
```

### Maxpool
```json
{
    "$type": "maxpool",
    "pool_size": 2
}
```

### Softmax
```json
{
    "$type": "softmax",
    "neurons": 10
}
```

## Optimizers

### AdaDelta

### AdaGrad

### Adam

### RmsProp

### Sgd

### SgdMomentum

## Learning rate schedulers

### Constant

### ExponentialDecay

### StepDecay

## Loss functions

### CrossEntropyLoss

### Mse