using ILGPU.Runtime;
using ILGPU;
using vcortex.Accelerated;

namespace vcortex.Layers;

public interface ILayer
{
    public int NumInputs { get; }
    public int NumOutputs { get; }

    public int GradientCount { get; }

    public int ActivationInputOffset { get; }
    public int ActivationOutputOffset { get; }
    public int CurrentLayerErrorOffset { get; }
    public int NextLayerErrorOffset { get; }
    public int GradientOffset { get; }
    public int ParameterCount { get; }
    public int ParameterOffset { get; }
    public float[] Parameters { get; set; }

    public void Forward(float[] activations);

    public virtual void FillRandom(NetworkAccelerator accelerator)
    {
    }

    public void Backward(float[] activations, float[] errors,
        float[] gradients, float learningRate);

    public void AccumulateGradients(float[][] gradients, float learningRate);

    public void Forward(NetworkAccelerator accelerator);
    public void Backward(NetworkAccelerator accelerator);
    public void AccumulateGradients(NetworkAccelerator accelerator);

    public void CompileKernels(Accelerator accelerator);

    public LayerData LayerData { get; }

}