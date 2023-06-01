using Camera.MAUI;
using Microsoft.ML;
using ONNXConsolePort.DataStructures;
using ONNXConsolePort.Drawables;
using ONNXConsolePort.YoloParser;
using System.Diagnostics;

namespace ONNXConsolePort;

public partial class LiveObjectDetectionPage : ContentPage
{
    private bool _isProcessing = false;
    private bool _isCameraPlaying = false;
    private IDispatcherTimer _timer = null;
    private string _assetsPath = FileSystem.AppDataDirectory;
    private string _outputDirectory;
    private string _outputFilePath;

    #region Life cycles and class construction

    public LiveObjectDetectionPage()
    {
        InitializeComponent();
        _outputDirectory = Path.Combine(_assetsPath, "snapshot");
        _outputFilePath = Path.Combine(_outputDirectory, "snapshot.png");
    }

    protected override void OnAppearing()
    {
        base.OnAppearing();

        cameraView.CamerasLoaded += CameraView_CamerasLoaded;
    }

    protected override void OnDisappearing()
    {
        base.OnDisappearing();

        cameraView.CamerasLoaded -= CameraView_CamerasLoaded;
    }

    #endregion

    #region Camera related

    private void CameraView_CamerasLoaded(object sender, EventArgs e)
    {
        if (cameraView.Cameras.Count() > 0)
        {
            cameraView.Camera = cameraView.Cameras.First();
            cameraButton.Text = "Start";
            snapShotButton.Text = "Start Snapshot";
            _isCameraPlaying = false;
        }
    }

    private void cameraButton_Clicked(object sender, EventArgs e)
    {
        if (_isCameraPlaying == false)
        {
            StartCamera();
        }
        else
        {
            StopCamera();
        }

    }

    private async void snapShotButton_Clicked(object sender, EventArgs e)
    {

        if (_timer == null || _timer.IsRunning == false)
        {
            await InitialiseFilesAsync();
            StartAutoSnapShotForProcessing();
            snapShotButton.Text = "Stop Snapshots";
        }
        else
        {
            StopAutoSnapShotForProcessing();
            snapShotButton.Text = "Start Snapshots";
        }
    }

    private void StopCamera()
    {
        MainThread.BeginInvokeOnMainThread(async () =>
        {
            if (await cameraView.StopCameraAsync() == CameraResult.Success)
            {
                _isCameraPlaying = false;
                cameraButton.Text = "Start";
            }
        });
    }

    private void StartCamera()
    {
        MainThread.BeginInvokeOnMainThread(async () =>
        {
            await cameraView.StopCameraAsync(); // a bug fix for some Samsung devices

            // Enable updating of snap shot
            cameraView.TakeAutoSnapShot = true;
            cameraView.AutoSnapShotSeconds = 1;
            cameraView.AutoSnapShotAsImageSource = true;

            if (await cameraView.StartCameraAsync() == CameraResult.Success)
            {
                cameraButton.Text = "Stop";
                _isCameraPlaying = true;
            }

        });


    }

    private void StartAutoSnapShotForProcessing()
    {
        if (_timer == default)
        {
            _timer = Application.Current.Dispatcher.CreateTimer();
            _timer.Interval = TimeSpan.FromSeconds(1);
            _timer.Tick += _timer_Tick; ;
            _timer.Start();
        }
    }

    private void _timer_Tick(object sender, EventArgs e)
    {
        // Run on main thread (UI thread) because we're updaing UI elements from the background thread
        MainThread.BeginInvokeOnMainThread(async () =>
        {
            if(_isProcessing == false)
            {
                _isProcessing = true;
                await StoreSnapshotAsync();
                DoObjectDetection();
                _isProcessing = false;
            }
        });
    }

    private Task StoreSnapshotAsync ()
    {
        Debug.WriteLine($"Storing image to {_outputFilePath}");
        if(Directory.Exists(_outputDirectory) == false )
        {
            Directory.CreateDirectory(_outputDirectory);
        }
        return cameraView.SaveSnapShot(Camera.MAUI.ImageFormat.PNG, _outputFilePath);
    }

    private void StopAutoSnapShotForProcessing()
    {
        _timer?.Stop();
        _timer = null;
    }

    #endregion

    #region OD Related

    private async Task InitialiseFilesAsync()
    {
        string exportedModelName = "model.onnx";
        if (CheckFileExists(exportedModelName) == false)
        {
            await MoveAssetFileToDataDirectory($"assets/Model/{exportedModelName}", $"{exportedModelName}");
        }

    }

    private bool CheckFileExists(string fileName)
    {
        return File.Exists(Path.Combine(FileSystem.Current.AppDataDirectory, fileName));
    }

    public async Task MoveAssetFileToDataDirectory(string sourceFile, string targetFileName)
    {
        // Read the source file -- note the use of 'using'
        using Stream readFileStream = await FileSystem.Current.OpenAppPackageFileAsync(sourceFile);

        // Write the file content to the app data directory
        string targetFile = Path.Combine(FileSystem.Current.AppDataDirectory, targetFileName);

        using FileStream outputStream = File.OpenWrite(targetFile);
        using StreamWriter streamWriter = new StreamWriter(outputStream);

        await readFileStream.CopyToAsync(streamWriter.BaseStream);
    }

    private void DoObjectDetection()
    {
        var modelFilePath = Path.Combine(_assetsPath, "model.onnx");

        MLContext mlContext = new MLContext();

        try
        {
            IEnumerable<ImageNetData> images = ImageNetData.ReadFromFile(_outputDirectory);
            IDataView imageDataView = mlContext.Data.LoadFromEnumerable(images);

            // Create instance of model scorer
            var modelScorer = new ExportedModelScorer(_outputDirectory, modelFilePath, mlContext);

            // Use model to score data
            IEnumerable<float[]> probabilities = modelScorer.Score(imageDataView);

            // Post processing
            ExportedYoloOutputParser parser = new ExportedYoloOutputParser();

            var probabilityMap = probabilities.First();

            var boundingBoxes = parser.ParseOutputs(probabilityMap);

            if (boundingBoxes.Any())
            {
                IList<YoloBoundingBox> detectedObjects = parser.FilterBoundingBoxes(boundingBoxes, 5, 0.5F);

                var filteredBoundingBoxes = parser.FilterBoundingBoxes(detectedObjects, 5, 0.5F);

                string imageFileName = images.First().Label;

                RenderProcessedImage(detectedObjects);
                LogDetectedObjects(imageFileName, filteredBoundingBoxes);
            }

        }
        catch (Exception ex)
        {
            Debug.WriteLine(ex.Message);
        }
    }

    void LogDetectedObjects(string imageName, IList<YoloBoundingBox> boundingBoxes)
    {
        Console.WriteLine($".....The objects in the image {imageName} are detected as below....");

        foreach (var box in boundingBoxes)
        {
            Console.WriteLine($"{box.Label} and its Confidence score: {box.Confidence}");
        }

        Console.WriteLine("");
    }

    void RenderProcessedImage(IList<YoloBoundingBox> filteredBoundingBoxes)
    {
        ProcessedImage.Drawable = new ProcessedImageFileDrawable()
        {
            ImagePath=_outputFilePath,
            BoundingBoxes = filteredBoundingBoxes
        };

        ProcessedImage.Invalidate();

    }

    #endregion

}