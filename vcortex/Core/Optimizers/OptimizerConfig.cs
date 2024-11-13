namespace vcortex.Core.Optimizers;

public abstract class OptimizerConfig
{
    
}

public class AdaDelta : OptimizerConfig
{
    public float Rho { get; set; } = 0.1f;
    public float Epsilon { get; set; } = 1e-8f;
}

public class AdaGrad : OptimizerConfig
{
    public float Epsilon { get; set; } = 1e-8f;
}

public class Adam : OptimizerConfig
{
    public float Beta1 { get; set; } = 0.9f;
    public float Beta2 { get; set; } = 0.999f;
    public float Epsilon { get; set; } = 1e-8f;
}

public class RmsProp : OptimizerConfig
{
    public float Rho { get; set; } = 0.1f;
    public float Epsilon { get; set; } = 1e-8f;
}

public class Sgd : OptimizerConfig
{
    
}
public class SgdMomentum : OptimizerConfig
{
    public float Momentum { get; set; } = 0.1f;
}
