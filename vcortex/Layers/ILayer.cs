namespace vcortex.Layers;

public interface ILayer
{
    public Layer Config { get; }
    public void FillRandom();
    public void Forward();
    public void Backward();
    public bool IsTraining { get; set; }
}