using vcortex.Layers.Convolution;

namespace vcortex;

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
    public float LearningRate { get; set; }
    public int Outputs { get; set; }
    public InputDateType InputDateType { get; set; }
    public IInputConfig InputConfig { get; set; }
}