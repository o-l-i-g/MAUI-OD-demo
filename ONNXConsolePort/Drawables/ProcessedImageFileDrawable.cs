using Microsoft.Maui.Graphics.Platform;
using ONNXConsolePort.YoloParser;
using System.Reflection;

namespace ONNXConsolePort.Drawables;

internal class ProcessedImageFileDrawable : BindableObject, IDrawable
{
    public string ImagePath { get; set; }
    public IList<YoloBoundingBox> BoundingBoxes { get; set; } = new List<YoloBoundingBox>();

    public void Draw(ICanvas canvas, RectF dirtyRect)
    {
        Microsoft.Maui.Graphics.IImage image;
        Assembly assembly = GetType().GetTypeInfo().Assembly;
        if (string.IsNullOrWhiteSpace(ImagePath))
        {
            return;
        }

        if(BoundingBoxes == default || BoundingBoxes.Any() == false)
        {
            return;
        }

        using (Stream stream = new FileStream(ImagePath, FileMode.Open))
        {
            image = PlatformImage.FromStream(stream); // doesn't work on windows
        }

        if (image != null)
        {
            var originalImageHeight = image.Height;
            var originalImageWidth = image.Width;
            var targetImageHeight = 200;
            var targetImageWidth = 200;
            
            foreach (var box in BoundingBoxes)
            {
                // make sure bounding box dimensions are within original image dimensions
                var x = (uint)Math.Max(box.Dimensions.X, 0);
                var y = (uint)Math.Max(box.Dimensions.Y, 0);
                var width = (uint)Math.Min(originalImageWidth - x, box.Dimensions.Width);
                var height = (uint)Math.Min(originalImageHeight - y, box.Dimensions.Height);

                // convert position and width based on ONNX output size
                x = (uint)originalImageWidth * x / OnnxModelScorer.ImageNetSettings.imageWidth;
                y = (uint)originalImageHeight * y / OnnxModelScorer.ImageNetSettings.imageHeight;
                width = (uint)originalImageWidth * width / OnnxModelScorer.ImageNetSettings.imageWidth;
                height = (uint)originalImageHeight * height / OnnxModelScorer.ImageNetSettings.imageHeight;

                // convert position and width based to new size (target size)
                float ratioX = targetImageWidth / originalImageWidth;
                float ratioY = targetImageHeight / originalImageHeight;

                float scaledX = x * ratioX;
                float scaledY = y * ratioY;
                float scaledWidth = width * ratioX;
                float scaledHeight = height * ratioY;

                string text = $"{box.Label} ({(box.Confidence * 100).ToString("0")}%)";

                // Define Text Options
                canvas.FontSize = 12;
                canvas.FontColor = Colors.Red;

                // Define BoundingBox options
                canvas.StrokeColor = Colors.Red;
                canvas.StrokeSize = 4;

                var newImage = image.Resize(targetImageWidth, targetImageHeight, ResizeMode.Stretch, disposeOriginal: false);
                canvas.DrawImage(newImage, 0, 0, newImage.Width, newImage.Height);

                canvas.DrawRectangle(scaledX, scaledY, scaledWidth, scaledHeight);

                canvas.DrawString(box.Label, scaledX, scaledY, HorizontalAlignment.Left);

            }
            
        }

    }
}
