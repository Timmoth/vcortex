namespace vcortex.Network;

public interface INetworkTrainerAgent : INetworkAgent
{
    public void Train(List<(float[] imageData, float[] label)> data);

    public void Test(List<(float[] imageData, float[] label)> data,
        float threshold);
}

public interface INetworkInferenceAgent : INetworkAgent
{
    public List<float[]> Predict(List<float[]> batches);
}