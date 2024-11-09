namespace vcortex.Layers;

public interface ILayer
{
    public int NumInputs { get; }
    public int NumOutputs { get; }

    public int GradientCount { get; }

    void Forward(float[] inputs, float[] outputs);

    public virtual void FillRandom()
    {
    }

    public void Backward(float[] inputs, float[] outputs, float[] currentLayerErrors, float[] nextLayerErrors,
        float[] gradients, float learningRate);

    public void AccumulateGradients(float[][] gradients, float learningRate);
}