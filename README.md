# vcortex
Lightweight CPU / GPU machine learning for dotnet

## Image Classification Quickstart
Quick guide to setting up a neural network for image classification.

```csharp
// Define the input structure
var inputConfig = new ConvolutionInputConfig
{
    Width = 28,         // Image width in pixels
    Height = 28,        // Image height in pixels
    Grayscale = true    // True for grayscale images, false for RGB
};

// Define your networks architecture
var network = new NetworkBuilder(inputConfig)
    .Add(new Convolution
    {
        Activation = ActivationType.LeakyRelu,    // Activation function
        KernelSize = 3,                           // Width/height of convolutional filter
        KernelsPerChannel = 32,                   // Number of filters per input channel
        Padding = 1,                              // Padding added to image borders
        Stride = 1                                // Step size of the filter
    })
    .Add(new Maxpool
    {
        PoolSize = 2                              // Size of the pooling window
    })
    .Add(new Dense
    {
        Activation = ActivationType.LeakyRelu,    // Activation function for dense layer
        Neurons = 128                             // Number of neurons in dense layer
    })
    .Add(new Dropout
    {
        DropoutRate = 0.2f                        // Dropout rate for regularization
    })
    .Add(new Softmax
    {
        Neurons = 10                              // Number of output classes
    })
    .Build();

// Define the training parameters   
var trainingConfig = new TrainConfig
{
    Epochs = 20,                                  // Total training iterations over dataset
    Scheduler = new ExponentialDecay()
    {
        InitialLearningRate = 0.001f,             // Starting learning rate
        DecayRate = 0.05f                         // Rate at which learning rate decays
    },
    Optimizer = new Adam(),                       // Optimization algorithm
    LossFunction = LossFunction.Mse,              // Loss function to minimize
    BatchSize = 100                               // Number of samples per training batch
};

// Create a CPU trainer
var trainer = new CpuNetworkTrainer(network, trainingConfig);
// Or create a GPU trainer
var trainer = new GpuNetworkTrainer(GpuType.Cuda, 0, net, trainingConfig);

// Initialize the trainable parameters to random values
trainer.InitRandomParameters();                    // Randomize model parameters
// Train
trainer.Train(trainData);                          // Train model on training data
// Test
trainer.Test(testData, 0.1f);                      // Test model on test data
```

## Persistence
Save and load network architecture and parameters.

```csharp
// Load the network architecture from disk
var network = NetworkConfig.DeserializeFromDisk("./network.json");
// Save the network architecture to disk
network.SerializeToDisk("./network.json");

// Load the network parameters from disk
trainer.SaveParametersToDisk("./weights.bin");
// Save the network parameters to disk
trainer.ReadParametersFromDisk("./weights.bin");

// Load the network parameters into an array
float[] parameters = trainer.GetParameters();  // Retrieve network parameters as array
// Load the network parameters from an array
trainer.LoadParameters(parameters);            // Load parameters from an array
```

## Layers
Common neural network layer configurations.

### Convolution
```
{
    "$type": "convolution",           # Specifies layer type as convolutional
    "stride": 1,                      # Filter movement per step
    "padding": 1,                     # Padding pixels around the image border
    "kernels_per_channel": 32,        # Number of filters per input channel
    "kernel_size": 3,                 # Size of each filter (e.g., 3x3)
    "activation": 2                   # Activation function type
}
```

### Dense
```
{
    "$type": "dense",                 # Specifies layer type as dense
    "activation": 2,                  # Activation function type
    "neurons": 128                    # Number of neurons in the dense layer
}
```

### Dropout
```
{
    "$type": "dropout",               # Specifies layer type as dropout
    "dropout_rate": 0.2               # Fraction of units to drop
}
```

### Maxpool
```
{
    "$type": "maxpool",               # Specifies layer type as max pooling
    "pool_size": 2                    # Size of the pooling filter
}
```

### Softmax
```
{
    "$type": "softmax",               # Specifies layer type as softmax for output
    "neurons": 10                     # Number of output classes
}
```

## Optimizers
Configurable algorithms for optimizing model weights.

### AdaDelta
```
{
    "$type": "adadelta",              # AdaDelta optimizer type
    "rho": 0.1,                       # Decay rate for squared gradient
    "epsilon": 1E-08                  # Small constant for numerical stability
}
```

### AdaGrad
```
{
    "$type": "adagrad",               # AdaGrad optimizer type
    "epsilon": 1E-08                  # Small constant for numerical stability
}
```

### Adam
```
{
    "$type": "adam",                  # Adam optimizer type
    "beta1": 0.9,                     # Exponential decay rate for first moment
    "beta2": 0.999,                   # Exponential decay rate for second moment
    "epsilon": 1E-08                  # Small constant for numerical stability
}
```

### RmsProp
```
{
    "$type": "rmsprop",               # RMSProp optimizer type
    "rho": 0.1,                       # Decay rate for moving average of squared gradient
    "epsilon": 1E-08                  # Small constant for numerical stability
}
```

### Sgd
```
{
    "$type": "sgd"                    # Stochastic Gradient Descent optimizer
}
```

### SgdMomentum
```
{
    "$type": "sgdmomentum",           # SGD with momentum optimizer type
    "momentum": 0.1                   # Momentum factor to accelerate SGD
}
```

## Learning rate schedulers
Adjust learning rate dynamically during training.

### Constant
```
{
    "$type": "constant",              # Constant learning rate scheduler
    "lr": 0.01                        # Fixed learning rate value
}
```

### ExponentialDecay
```
{
    "$type": "exponential_decay",     # Exponential decay scheduler
    "lr": 0.01,                       # Initial learning rate
    "decay": 0.05                     # Decay factor applied per epoch
}
```

### StepDecay
```
{
    "$type": "step_decay",            # Step decay scheduler
    "lr": 0.01,                       # Initial learning rate
    "step": 10,                       # Epoch interval for decay
    "decay": 0.5                      # Factor by which to decrease learning rate
}
```
## Loss functions
Functions to calculate error between predicted and true labels.

### CrossEntropyLoss
Used for classification tasks; compares class probabilities.

### Mse
Mean Squared Error for regression tasks; measures prediction accuracy.
