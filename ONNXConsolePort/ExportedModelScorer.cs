using Microsoft.ML;
using Microsoft.ML.Data;
using ONNXConsolePort.DataStructures;

namespace ONNXConsolePort;

public class ExportedModelScorer
{
    private readonly string imagesFolder;
    private readonly string modelLocation;
    private readonly MLContext mlContext;

    public struct ImageNetSettings
    {
        public const int imageHeight = 416;
        public const int imageWidth = 416;
    }

    public struct ExportedModelSettings
    {
        // for checking Tiny yolo2 Model input and  output  parameter names,
        //you can use tools like Netron, 
        // which is installed by Visual Studio AI Tools

        // input tensor name
        public const string ModelInput = "data";

        // output tensor name
        public const string ModelOutput = "model_outputs0";
    }

    public ExportedModelScorer(string imagesFolder, string modelLocation, MLContext mlContext)
    {
        this.imagesFolder = imagesFolder;
        this.modelLocation = modelLocation;
        this.mlContext = mlContext;
    }

    private ITransformer LoadModel(string modelLocation)
    {
        Console.WriteLine("Read model");
        Console.WriteLine($"Model location: {modelLocation}");
        Console.WriteLine($"Default parameters: image size=({ImageNetSettings.imageWidth},{ImageNetSettings.imageHeight})");
        var data = mlContext.Data.LoadFromEnumerable(new List<ImageNetData>());
        var pipeline = mlContext.Transforms.LoadImages(outputColumnName: "image", imageFolder: "", inputColumnName: nameof(ImageNetData.ImagePath))
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "resized_image", inputColumnName: "image", imageWidth: ImageNetSettings.imageWidth, imageHeight: ImageNetSettings.imageHeight))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "data", inputColumnName: "resized_image"))
                .Append(mlContext.Transforms.ApplyOnnxModel(modelFile: modelLocation, outputColumnNames: new[] { ExportedModelSettings.ModelOutput }, inputColumnNames: new[] { ExportedModelSettings.ModelInput }));

        // uncomment and put a breakpoint here to see preview of transform columns during Debug.
        //var preview = pipeline.Fit(data).Transform(data).Preview();

        var model = pipeline.Fit(data);

        return model;
    }

    private IEnumerable<float[]> PredictDataUsingModel(IDataView testData, ITransformer model)
    {
        Console.WriteLine($"Images location: {imagesFolder}");
        Console.WriteLine("");
        Console.WriteLine("=====Identify the objects in the images=====");
        Console.WriteLine("");

        IDataView scoredData = model.Transform(testData);

        IEnumerable<float[]> probabilities = scoredData.GetColumn<float[]>(ExportedModelSettings.ModelOutput);

        return probabilities;
    }

    public IEnumerable<float[]> Score(IDataView data)
    {
        var model = LoadModel(modelLocation);

        return PredictDataUsingModel(data, model);
    }

}


