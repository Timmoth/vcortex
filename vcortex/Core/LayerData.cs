namespace vcortex.Core;

public readonly struct LayerData
{
    public readonly int NumInputs;
    public readonly int NumOutputs;
    public readonly int ActivationInputOffset;
    public readonly int ActivationOutputOffset;
    public readonly int NextLayerErrorOffset;
    public readonly int CurrentLayerErrorOffset;
    public readonly int ParameterOffset;
    public readonly int BiasOffset;
    public readonly int InputWidth;
    public readonly int InputHeight;
    public readonly int OutputWidth;
    public readonly int OutputHeight;
    public readonly int InputChannels;
    public readonly int OutputChannels;
    public readonly int NumKernels;
    public readonly int KernelSize;
    public readonly int PoolSize;
    public readonly int Stride;
    public readonly int Padding;

    public LayerData(int numInputs, int numOutputs, int activationInputOffset, int activationOutputOffset, int nextLayerErrorOffset, int currentLayerErrorOffset, int parameterOffset, int biasOffset,
        int inputWidth, int inputHeight, int outputWidth, int outputHeight, int inputChannels, int outputChannels,
        int numKernels, int kernelSize, int poolSize, int stride= 0, int padding = 0)
    {
        NumInputs = numInputs;
        NumOutputs = numOutputs;
        ActivationInputOffset = activationInputOffset;
        ActivationOutputOffset = activationOutputOffset;
        NextLayerErrorOffset = nextLayerErrorOffset;
        CurrentLayerErrorOffset = currentLayerErrorOffset;
        ParameterOffset = parameterOffset;
        BiasOffset = biasOffset;
        InputWidth = inputWidth;
        InputHeight = inputHeight;
        OutputWidth = outputWidth;
        OutputHeight = outputHeight;
        InputChannels = inputChannels;
        OutputChannels = outputChannels;
        NumKernels = numKernels;
        KernelSize = kernelSize;
        PoolSize = poolSize;
        Stride = stride;
        Padding = padding;
    }
}