namespace vcortex.Accelerated;

public struct LayerData
{
    public int NumOutputs { get; set; }
    public int ParameterOffset { get; set; }
    public int BiasOffset { get; set; }
    public int NumInputs { get; set; }
    public int ActivationInputOffset { get; set; }
    public int ActivationOutputOffset { get; set; }
    public int GradientOffset { get; set; }
    public int NextLayerErrorOffset { get; set; }
    public int CurrentLayerErrorOffset { get; set; }
    public int InputWidth { get; set; }
    public int InputHeight { get; set; }
    public int OutputWidth { get; set; }
    public int OutputHeight { get; set; }
    public int NumKernels { get; set; }
    public int InputChannels { get; set; }
    public int KernelSize { get; set; }
    public int OutputChannels { get; set; }
    public int PoolSize { get; set; }
}