using vcortex.Input;
using vcortex.LearningRate;
using vcortex.Optimizers;

namespace vcortex.Training;

public enum InputDateType
{
    Csv,
    Directory
}

public class TrainConfig
{
    public string TrainPath { get; set; }
    public string TestPath { get; set; }
    public int Epochs { get; set; }
    public int Outputs { get; set; }
    public InputDateType InputDateType { get; set; }
    public IInputConfig InputConfig { get; set; }
    public LearningRateScheduler Scheduler { get; set; }
    public OptimizerConfig Optimizer { get; set; }
}