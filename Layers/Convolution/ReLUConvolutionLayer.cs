//namespace vcortex.Layers.Convolution;

//public class ReLUConvolutionLayer : IConvolutionalLayer
//{
//    public int NumInputs => InputWidth * InputHeight * InputChannels;
//    public int NumOutputs => InputWidth * InputHeight * InputChannels;

//    public int InputWidth { get; private set; }
//    public int InputHeight { get; private set; }
//    public int OutputWidth { get; private set; }
//    public int OutputHeight { get; private set; }
//    public int InputChannels { get; private set; }
//    public int OutputChannels { get; private set; }

//    public void Connect(IConvolutionalLayer prevLayer)
//    {
//        OutputWidth = InputWidth = prevLayer.OutputWidth;
//        OutputHeight = InputHeight = prevLayer.OutputHeight;
//        OutputChannels = InputChannels = prevLayer.OutputChannels;
//    }

//    public void Connect(ConvolutionInputConfig config)
//    {
//        OutputWidth = InputWidth = config.Width;
//        OutputHeight = InputHeight = config.Height;
//        OutputChannels = InputChannels = config.Grayscale ? 1 : 3;
//    }

//    public void Forward(float[] inputs, float[] outputs)
//    {
//        for (var i = 0; i < NumOutputs; i++)
//            // Apply ReLU activation
//            outputs[i] = Math.Max(0, inputs[i]);
//    }

//    public int GradientCount => 0;

//    public void Backward(float[] inputs, float[] outputs, float[] currentLayerErrors, float[] nextLayerErrors,
//        float[] gradients,
//        float learningRate)
//    {
//        Array.Clear(currentLayerErrors, 0, currentLayerErrors.Length);

//        for (var i = 0; i < NumOutputs; i++)
//            // Backpropagate only for positive outputs since ReLU's gradient is 0 for inputs < 0
//            currentLayerErrors[i] = inputs[i] > 0 ? nextLayerErrors[i] : 0;
//    }

//    public void AccumulateGradients(float[][] gradients, float learningRate)
//    {
//    }
//}