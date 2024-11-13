namespace vcortex.Core;

public readonly struct NetworkData
{
    public readonly int ActivationCount;
    public readonly int ParameterCount;

    public NetworkData(int activationCount, int parameterCount)
    {
        ActivationCount = activationCount;
        ParameterCount = parameterCount;
    }
}