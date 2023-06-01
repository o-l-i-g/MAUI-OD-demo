//using Camera.MAUI;
using Camera.MAUI;
using System.ComponentModel;

namespace ONNXConsolePort;

public partial class CameraPage : ContentPage
{
    private bool isPlaying = false;
    private IDispatcherTimer _timer = null;
    private string _outputPath = string.Empty;

    public CameraPage()
	{
		InitializeComponent();
        BindingContext = this;

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

    private void CameraView_CamerasLoaded(object sender, EventArgs e)
    {
        if(cameraView.NumCamerasDetected > 0)
        {
            cameraView.Camera = cameraView.Cameras.First();
            cameraButton.Text = "Start";
            snapShotButton.Text = "Start Snapshot";
            isPlaying = false;
        }
    }

    private void cameraButton_Clicked(object sender, EventArgs e)
    {
        if (isPlaying == false)
        {
            StartCamera();
        }
        else
        {
            StopCamera();
        }

    }

    private void snapShotButton_Clicked(object sender, EventArgs e)
    {
        if(_timer == null || _timer.IsRunning == false)
        {
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
            if(await cameraView.StopCameraAsync() == CameraResult.Success)
            {
                isPlaying = false;
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
                isPlaying=true;
            }

        });

        
    }

    private void StartAutoSnapShotForProcessing()
    {
        if( _timer == default)
        {
            _timer = Application.Current.Dispatcher.CreateTimer();
            _timer.Interval = TimeSpan.FromSeconds(2);
            _timer.Tick += _timer_Tick; ;
            _timer.Start();
        }
    }

    private void _timer_Tick(object sender, EventArgs e)
    {
        // Run on main thread (UI thread) because we're updaing UI elements from the background thread
        MainThread.BeginInvokeOnMainThread(() =>
        {
            TakeSnapshot();
        });
    }

    private void TakeSnapshot()
    {
        processedImage.Source = this.cameraView.SnapShot;

        // If you want to save as a file:
        //var image = cameraView.SaveSnapShot(Camera.MAUI.ImageFormat.PNG, _outputPath); 
    }

    private void StopAutoSnapShotForProcessing()
    {
        _timer?.Stop();
        _timer = null;
    }
}