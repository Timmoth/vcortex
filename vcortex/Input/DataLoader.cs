using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using vcortex.Layers;

namespace vcortex.Input;

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


    public static (List<(float[] imageData, float[] label)> train, List<(float[] imageData, float[] label)> test)
        LoadData(ConvolutionInputConfig inputConfig, string trainPath, string testPath)
    {
        var trainImages = LoadImages(trainPath, inputConfig);
        var testImages = LoadImages(testPath, inputConfig);
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


    public static List<(float[] inputs, float[] outputs)> LoadCsv(string filePath,
        ConvolutionInputConfig inputConfig, int outputs)
    {
        var result = new List<(float[] inputs, float[] outputs)>();

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
            var labels = new float[outputs];
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