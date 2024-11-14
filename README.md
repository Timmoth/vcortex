# vcortex
Lightweight CPU/GPU machine learning library for .NET, designed for neural network training and inference.

## Image Classification Quickstart
Quick guide to setting up and training a neural network for image classification tasks.

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
Configuration for convolutional layer used for feature extraction in image data.


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
Configuration for a fully connected (dense) layer used to combine features from previous layers.

```
{
    "$type": "dense",                 # Specifies layer type as dense
    "activation": 2,                  # Activation function type
    "neurons": 128                    # Number of neurons in the dense layer
}
```

### Dropout
Configuration for dropout layer used for regularization to prevent overfitting.

```
{
    "$type": "dropout",               # Specifies layer type as dropout
    "dropout_rate": 0.2               # Fraction of units to drop
}
```

### Maxpool
Configuration for max pooling layer used to reduce spatial dimensions.

```
{
    "$type": "maxpool",               # Specifies layer type as max pooling
    "pool_size": 2                    # Size of the pooling filter
}
```

### Softmax
Configuration for softmax layer used in the output for classification tasks.

```
{
    "$type": "softmax",               # Specifies layer type as softmax for output
    "neurons": 10                     # Number of output classes
}
```

## Optimizers
Configurable algorithms for optimizing model weights.

### AdaDelta
Optimizer that adapts learning rate based on a moving window of gradient updates.

```
{
    "$type": "adadelta",              # AdaDelta optimizer type
    "rho": 0.1,                       # Decay rate for squared gradient
    "epsilon": 1E-08                  # Small constant for numerical stability
}
```

### AdaGrad
Optimizer that adapts learning rates for each parameter based on past gradients.

```
{
    "$type": "adagrad",               # AdaGrad optimizer type
    "epsilon": 1E-08                  # Small constant for numerical stability
}
```

### Adam
Popular optimizer that combines momentum and adaptive learning rates for fast convergence.

```
{
    "$type": "adam",                  # Adam optimizer type
    "beta1": 0.9,                     # Exponential decay rate for first moment
    "beta2": 0.999,                   # Exponential decay rate for second moment
    "epsilon": 1E-08                  # Small constant for numerical stability
}
```

### RmsProp
Optimizer that adjusts learning rate by dividing by a running average of gradients.

```
{
    "$type": "rmsprop",               # RMSProp optimizer type
    "rho": 0.1,                       # Decay rate for moving average of squared gradient
    "epsilon": 1E-08                  # Small constant for numerical stability
}
```

### Sgd
Stochastic Gradient Descent, a traditional optimizer using a fixed learning rate.

```
{
    "$type": "sgd"                    # Stochastic Gradient Descent optimizer
}
```

### SgdMomentum
SGD variant that incorporates momentum to accelerate convergence.

```
{
    "$type": "sgdmomentum",           # SGD with momentum optimizer type
    "momentum": 0.1                   # Momentum factor to accelerate SGD
}
```

## Learning rate schedulers
Adjust learning rate dynamically during training.

### Constant
Scheduler with a fixed learning rate across all training epochs.

```
{
    "$type": "constant",              # Constant learning rate scheduler
    "lr": 0.01                        # Fixed learning rate value
}
```

### ExponentialDecay
Scheduler that gradually decreases learning rate exponentially over time.

```
{
    "$type": "exponential_decay",     # Exponential decay scheduler
    "lr": 0.01,                       # Initial learning rate
    "decay": 0.05                     # Decay factor applied per epoch
}
```

### StepDecay
Scheduler that reduces the learning rate at set intervals during training.

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
Loss function used in classification tasks; compares model's predicted probabilities to true class labels.

### Mse
Mean Squared Error loss function used in regression tasks; measures accuracy by averaging squared prediction errors.

### Training Config
Defines the training parameters and configuration for optimizing the neural network.

```
{
  "epochs": 20,                     # Total number of training epochs or iterations over the dataset
  "lr_schedule": {                  
    "$type": "exponential_decay",   # Type of learning rate scheduler (e.g., exponential decay)
    "lr": 0.001,                    # Initial learning rate to be used at the start of training
    "decay": 0.05                   # Rate at which the learning rate decreases each epoch
  },
  "optimizer": {                    
    "$type": "adam",                # Optimization algorithm type (e.g., Adam)
    "beta1": 0.9,                   # Exponential decay rate for the first moment estimates (Adam)
    "beta2": 0.999,                 # Exponential decay rate for the second moment estimates (Adam)
    "epsilon": 1E-08                # Small constant for numerical stability in Adam optimizer
  },
  "loss": 0,                        # Loss function identifier (e.g., 0 for MSE, 1 for CrossEntropy)
  "batch": 100                      # Size of each batch of data samples during training
}

```