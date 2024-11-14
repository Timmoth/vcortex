using System.Text.Json.Serialization;

namespace vcortex.Optimizers;

[JsonPolymorphic]
[JsonDerivedType(typeof(AdaDelta), "adadelta")]
[JsonDerivedType(typeof(AdaGrad), "adagrad")]
[JsonDerivedType(typeof(Adam), "adam")]
[JsonDerivedType(typeof(RmsProp), "rmsprop")]
[JsonDerivedType(typeof(Sgd), "sgd")]
[JsonDerivedType(typeof(SgdMomentum), "sgdmomentum")]
public abstract class OptimizerConfig
{
}

public class AdaDelta : OptimizerConfig
{
    [JsonPropertyName("rho")]
    public float Rho { get; set; } = 0.1f;
    [JsonPropertyName("epsilon")]
    public float Epsilon { get; set; } = 1e-8f;
}

public class AdaGrad : OptimizerConfig
{
    [JsonPropertyName("epsilon")]
    public float Epsilon { get; set; } = 1e-8f;
}

public class Adam : OptimizerConfig
{
    [JsonPropertyName("beta1")]
    public float Beta1 { get; set; } = 0.9f;
    [JsonPropertyName("beta2")]
    public float Beta2 { get; set; } = 0.999f;
    [JsonPropertyName("epsilon")]
    public float Epsilon { get; set; } = 1e-8f;
}

public class RmsProp : OptimizerConfig
{
    [JsonPropertyName("rho")]
    public float Rho { get; set; } = 0.1f;
    [JsonPropertyName("epsilon")]
    public float Epsilon { get; set; } = 1e-8f;
}

public class Sgd : OptimizerConfig
{
}

public class SgdMomentum : OptimizerConfig
{
    [JsonPropertyName("momentum")]
    public float Momentum { get; set; } = 0.1f;
}