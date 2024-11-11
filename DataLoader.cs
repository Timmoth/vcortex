using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using vcortex.Layers.Convolution;

namespace vcortex;

public static class DataLoader
{
    public static float[] LoadImageRgb(string path, int width, int height)
    {
        using var image = Image.Load<Rgb24>(path);
        image.Mutate(ctx => ctx.Resize(width, height));
        Span<byte> pixels = stackalloc byte[width * height * 3];

        image.CopyPixelDataTo(pixels);
        var channelSize = width * height;

        var pixelData = new float[channelSize * 3];

        var channelIndex = 0;
        for (var i = 0; i < pixels.Length; i += 3)
        {
            pixelData[channelIndex] = pixels[i] / 255f;
            pixelData[channelIndex + channelSize] = pixels[i + 1] / 255f;
            pixelData[channelIndex + 2 * channelSize] = pixels[i + 2] / 255f;

            channelIndex++;
        }

        return pixelData;
    }

    public static void SaveImageRgb(string outputPath, int width, int height, Span<float> r, Span<float> g,
        Span<float> b)
    {
        Span<byte> pixels = stackalloc byte[width * height * 3];

        var offset = 0;
        for (var i = 0; i < width * height; i++)
        {
            pixels[offset] = (byte)(r[i] * 255f);
            pixels[offset + 1] = (byte)(g[i] * 255f);
            pixels[offset + 2] = (byte)(b[i] * 255f);
            offset += 3;
        }

        using var image = Image.LoadPixelData<Rgb24>(pixels, width, height);
        image.Save(outputPath);
    }

    public static float[] LoadImageGrayScale(string path, int width, int height)
    {
        using var image = Image.Load<Rgb24>(path);
        image.Mutate(ctx => ctx.Resize(width, height));
        Span<byte> pixels = stackalloc byte[width * height * 3];

        image.CopyPixelDataTo(pixels);
        var channelSize = width * height;

        var pixelData = new float[channelSize * 3];

        var channelIndex = 0;
        for (var i = 0; i < pixels.Length; i += 3)
            pixelData[channelIndex++] = (0.299f * pixels[i] + 0.587f * pixels[i + 1] + 0.114f * pixels[i + 2]) / 255f;

        return pixelData;
    }

    public static void SaveImageGrayScale(string outputPath, int width, int height, Span<float> pixelData)
    {
        Span<byte> pixels = stackalloc byte[width * height * 3];

        var offset = 0;
        for (var i = 0; i < width * height; i++)
        {
            pixels[offset] = (byte)(pixelData[i] * 255f);
            pixels[offset + 1] = (byte)(pixelData[i] * 255f);
            pixels[offset + 2] = (byte)(pixelData[i] * 255f);
            offset += 3;
        }

        using var image = Image.LoadPixelData<Rgb24>(pixels, width, height);
        image.Save(outputPath);
    }

    public static List<List<float[]>> LoadImages(string path, ConvolutionInputConfig config)
    {
        var allImages = new List<List<float[]>>();
        foreach (var dir in Directory.EnumerateDirectories(path))
        {
            var images = new List<float[]>();
            foreach (var file in Directory.EnumerateFiles($"{dir}"))
                images.Add(config.Grayscale
                    ? LoadImageGrayScale(file, config.Width, config.Height)
                    : LoadImageRgb(file, config.Width, config.Height));

            allImages.Add(images);
        }

        return allImages;
    }

    //public static void OutputImageRgb(KernelConvolutionLayer l1, float[] imageData, string prefix)
    //{
    //    var inputChannelSize = l1.InputWidth * l1.InputHeight;
    //    var img = imageData.AsSpan();
    //    SaveImageRgb($"./{prefix}_original.jpeg", l1.InputWidth, l1.InputHeight,
    //        img.Slice(0, inputChannelSize),
    //        img.Slice(1 * inputChannelSize, inputChannelSize),
    //        img.Slice(2 * inputChannelSize, inputChannelSize));

    //    var o = new float[l1.NumOutputs];
    //    l1.Forward(imageData, o);

    //    var offset = 0;
    //    var output = o.AsSpan();
    //    var channelSize = l1.OutputWidth * l1.OutputHeight;

    //    for (var i = 0; i < l1.NumKernels; i++)
    //    {
    //        SaveImageRgb($"./{prefix}_kernel_{i}.jpeg", l1.OutputWidth, l1.OutputHeight,
    //            output.Slice(offset * channelSize, channelSize),
    //            output.Slice((offset + 1) * channelSize, channelSize),
    //            output.Slice((offset + 2) * channelSize, channelSize));

    //        offset += 3;
    //    }
    //}

    //public static void OutputImageGrayScale(KernelConvolutionLayer l1, float[] imageData, string prefix)
    //{
    //    SaveImageGrayScale($"./{prefix}_original.jpeg", l1.InputWidth, l1.InputHeight,
    //        imageData);

    //    var o = new float[l1.NumOutputs];

    //    l1.Forward(imageData, o);

    //    var channelSize = l1.OutputWidth * l1.OutputHeight;
    //    var output = o.AsSpan();
    //    for (var i = 0; i < l1.NumKernels; i++)
    //        SaveImageGrayScale($"./{prefix}_kernel_{i}.jpeg", l1.OutputWidth, l1.OutputHeight,
    //            output.Slice(i * channelSize, channelSize));
    //}

    public static (List<(float[] imageData, float[] label)> train, List<(float[] imageData, float[] label)> test)
        LoadData(TrainConfig config)
    {
        var inputConfig = config.InputConfig as ConvolutionInputConfig;
        var trainImages = LoadImages(config.TrainPath, inputConfig);
        var testImages = LoadImages(config.TestPath, inputConfig);
        Console.WriteLine("Loaded {0} training and {1} testing images", trainImages.Sum(t => t.Count),
            testImages.Sum(t => t.Count));

        var trainData = new List<(float[] imageData, float[] label)>();
        for (var index = 0; index < trainImages.Count; index++)
        {
            var output = new float[trainImages.Count];
            output[index] = 1.0f;

            var imageData = trainImages[index];
            foreach (var image in imageData) trainData.Add((image, output));
        }

        var testData = new List<(float[] imageData, float[] label)>();
        for (var index = 0; index < testImages.Count; index++)
        {
            var output = new float[testImages.Count];
            output[index] = 1.0f;

            var imageData = testImages[index];
            foreach (var image in imageData) testData.Add((image, output));
        }

        return (trainData, testData);
    }

    public static (List<(float[] imageData, float[] label)> train, List<(float[] imageData, float[] label)> test)
        LoadMNIST(TrainConfig config)
    {
        var inputConfig = config.InputConfig as ConvolutionInputConfig;
        var trainImages = LoadMNISTCsv(config.TrainPath, inputConfig);
        var testImages = LoadMNISTCsv(config.TestPath, inputConfig);
        Console.WriteLine("Loaded {0} training and {1} testing images", trainImages.Count, testImages.Count);

        return (trainImages, testImages);
    }

    private static List<(float[] imageData, float[] label)> LoadMNISTCsv(string filePath,
        ConvolutionInputConfig inputConfig)
    {
        var result = new List<(float[] imageData, float[] label)>();

        // Open the file for reading
        using var reader = new StreamReader(filePath);
        string? line;
        // Skip header if there's any
        reader.ReadLine();

        // Read each line one by one
        while ((line = reader.ReadLine()) != null)
        {
            // Split by commas (',') and convert the values
            var columns = line.Split(',');

            // First column is the label (image label)
            var labelIndex = byte.Parse(columns[0]);
            var labels = new float[10];
            labels[labelIndex] = 1.0f;

            // The rest are pixel values (image data)
            var imageData = new float[inputConfig.Width * inputConfig.Height];
            for (var i = 1; i < columns.Length; i++)
                // Convert string pixel values to byte and assign to imageData
                imageData[i - 1] = float.Parse(columns[i]) / 255f;

            // Add the tuple (imageData, label) to the result list
            result.Add((imageData, labels));
        }

        return result;
    }
}