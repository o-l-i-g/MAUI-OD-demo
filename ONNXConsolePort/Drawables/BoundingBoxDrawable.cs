using Microsoft.Maui.Graphics.Platform;
using ONNXConsolePort.YoloParser;
using System.Reflection;

namespace ONNXConsolePort.Drawables;

internal class BoundingBoxDrawable : BindableObject, IDrawable
{
    public double ImageHeight { get; set; }
    public double ImageWidth { get; set; }

    public IList<YoloBoundingBox> BoundingBoxes { get; set; } = new List<YoloBoundingBox>();

    public void Draw(ICanvas canvas, RectF dirtyRect)
    {
        Microsoft.Maui.Graphics.IImage image;

        foreach (var box in BoundingBoxes)
        {
            // make sure bounding box dimensions are within original image dimensions
            var x = (uint)Math.Max(box.Dimensions.X, 0);
            var y = (uint)Math.Max(box.Dimensions.Y, 0);
            var width = (uint)Math.Min(ImageWidth - x, box.Dimensions.Width);
            var height = (uint)Math.Min(ImageHeight - y, box.Dimensions.Height);

            // convert position and width based on ONNX output size
            x = (uint)ImageWidth * x / OnnxModelScorer.ImageNetSettings.imageWidth;
            y = (uint)ImageHeight * y / OnnxModelScorer.ImageNetSettings.imageHeight;
            width = (uint)ImageWidth * width / OnnxModelScorer.ImageNetSettings.imageWidth;
            height = (uint)ImageHeight * height / OnnxModelScorer.ImageNetSettings.imageHeight;

            string text = $"{box.Label} ({(box.Confidence * 100).ToString("0")}%)";

            // Define Text Options
            canvas.FontSize = 12;
            canvas.FontColor = Colors.Red;

            // Define BoundingBox options
            canvas.StrokeColor = Colors.Red;
            canvas.StrokeSize = 4;

            canvas.DrawRectangle(x, y, width, height);

            canvas.DrawString(box.Label, x, y, HorizontalAlignment.Left);

        }
            

    }
}
