//namespace vcortex.Layers.Convolution;

//public class LeakyReLUConvolutionLayer : IConvolutionalLayer
//{
//    private readonly float _alpha;

//    public LeakyReLUConvolutionLayer(float alpha = 0.01f)
//    {
//        _alpha = alpha; // Slope for negative inputs
//    }

//    public int NumInputs => InputWidth * InputHeight * InputChannels;
//    public int NumOutputs => InputWidth * InputHeight * InputChannels;

//    public int InputWidth { get; private set; }
//    public int InputHeight { get; private set; }
//    public int OutputWidth { get; private set; }
//    public int OutputHeight { get; private set; }
//    public int InputChannels { get; private set; }
//    public int OutputChannels { get; private set; }
//    public int ActivationInputOffset { get; private set; }
//    public int ActivationOutputOffset { get; private set; }
//    public int CurrentLayerErrorOffset { get; private set; }
//    public int NextLayerErrorOffset { get; private set; }
//    public int GradientOffset { get; private set; }
//    public int ParameterCount { get; private set; }
//    public int ParameterOffset { get; private set; }
//    public float[] Parameters { get; set; }

//    public void Connect(IConvolutionalLayer prevLayer)
//    {
//        OutputWidth = InputWidth = prevLayer.OutputWidth;
//        OutputHeight = InputHeight = prevLayer.OutputHeight;
//        OutputChannels = InputChannels = prevLayer.OutputChannels;

//        ParameterCount = 0;
//        ParameterOffset = prevLayer.ParameterOffset + prevLayer.ParameterCount;

//        ActivationInputOffset = prevLayer.ActivationOutputOffset;
//        ActivationOutputOffset = prevLayer.ActivationOutputOffset + prevLayer.NumOutputs;
//        CurrentLayerErrorOffset = prevLayer.NextLayerErrorOffset;
//        NextLayerErrorOffset = prevLayer.CurrentLayerErrorOffset + NumInputs;
//        GradientOffset = prevLayer.GradientOffset + prevLayer.GradientCount;
//    }

//    public void Connect(ConvolutionInputConfig config)
//    {
//        OutputWidth = InputWidth = config.Width;
//        OutputHeight = InputHeight = config.Height;
//        OutputChannels = InputChannels = config.Grayscale ? 1 : 3;

//        ParameterCount = 0;
//        ParameterOffset = 0;

//        ActivationInputOffset = 0;
//        ActivationOutputOffset = NumInputs;
//        CurrentLayerErrorOffset = 0;
//        NextLayerErrorOffset = NumInputs;
//        GradientOffset = 0;
//    }

//    public void Forward(float[] activations)
//    {
//        for (var i = 0; i < NumOutputs; i++)
//            // Apply Leaky ReLU activation
//            activations[ActivationOutputOffset + i] = activations[ActivationInputOffset + i] > 0 ? activations[ActivationInputOffset + i] : _alpha * activations[ActivationInputOffset + i];
//    }

//    public int GradientCount => 0;

//    public void Backward(float[] activations, float[] errors,
//        float[] gradients, float learningRate)
//    {
//        Array.Clear(errors, CurrentLayerErrorOffset, NumInputs);

//        for (var i = 0; i < NumOutputs; i++)
//            // Use _alpha as gradient for negative inputs
//            errors[CurrentLayerErrorOffset + i] = activations[ActivationInputOffset + i] > 0 ? errors[NextLayerErrorOffset + i] : _alpha * errors[NextLayerErrorOffset + i];
//    }

//    public void AccumulateGradients(float[][] gradients, float learningRate)
//    {
//    }
//}

