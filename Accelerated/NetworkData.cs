namespace vcortex.Accelerated;

public readonly struct NetworkData
{
    public readonly float LearningRate;
    public readonly int ActivationCount;
    public readonly int ErrorCount;
    public readonly int GradientCount;
    public readonly int BatchSize;
    public readonly float Beta1;
    public readonly float Beta2;
    public readonly float Epsilon;
    public readonly int Timestep;

    public NetworkData(float learningRate, int activationCount, int errorCount, int gradientCount, int batchSize,
        float beta1, float beta2, float epsilon, int timestep)
    {
        LearningRate = learningRate;
        ActivationCount = activationCount;
        ErrorCount = errorCount;
        GradientCount = gradientCount;
        BatchSize = batchSize;
        Beta1 = beta1;
        Beta2 = beta2;
        Epsilon = epsilon;
        Timestep = timestep;
    }
}

public static class NetworkDataExtensions
{
    public static NetworkData IncrementTimestep(this NetworkData networkData)
    {
        return new NetworkData(networkData.LearningRate, networkData.ActivationCount,
            networkData.ActivationCount, networkData.GradientCount, networkData.BatchSize, networkData.Beta1, networkData.Beta2, networkData.Epsilon,
            networkData.Timestep + 1);
    }
    
    public static NetworkData ResetTimestep(this NetworkData networkData)
    {
        return new NetworkData(networkData.LearningRate, networkData.ActivationCount,
            networkData.ActivationCount, networkData.GradientCount, networkData.BatchSize, networkData.Beta1, networkData.Beta2, networkData.Epsilon,
            0);
    }
}