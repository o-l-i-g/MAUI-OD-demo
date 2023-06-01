using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using Color = System.Drawing.Color;

namespace ONNXConsolePort.YoloParser;

public class ExportedYoloOutputParser
{
    public const int ROW_COUNT = 13;
    public const int COL_COUNT = 13;
    public const int CHANNEL_COUNT = 125;
    public const int BOXES_PER_CELL = 5;
    public const int BOX_INFO_FEATURE_COUNT = 5;
    public const int CLASS_COUNT = 1;
    public const float CELL_WIDTH = 32;
    public const float CELL_HEIGHT = 32;

    private int channelStride = ROW_COUNT * COL_COUNT;
    private float[] anchors = new float[]
{
    0.573f, 0.677f, 1.87f, 2.06f, 3.34f, 5.47f, 7.88f, 3.53f, 9.77f, 9.17f
};

    private string[] labels = new string[]
{
    "MS Bit"
};

    private static Color[] classColors = new Color[]
{
    Color.Khaki,
    Color.Fuchsia,
    Color.Silver,
    Color.RoyalBlue,
    Color.Green,
    Color.DarkOrange,
    Color.Purple,
    Color.Gold,
    Color.Red,
    Color.Aquamarine,
    Color.Lime,
    Color.AliceBlue,
    Color.Sienna,
    Color.Orchid,
    Color.Tan,
    Color.LightPink,
    Color.Yellow,
    Color.HotPink,
    Color.OliveDrab,
    Color.SandyBrown,
    Color.DarkTurquoise
};

    private float Sigmoid(float value)
    {
        var k = (float)Math.Exp(value);
        return k / (1.0f + k);
    }

    private float[] Softmax(float[] values)
    {
        var maxVal = values.Max();
        var exp = values.Select(v => Math.Exp(v - maxVal));
        var sumExp = exp.Sum();

        return exp.Select(v => (float)(v / sumExp)).ToArray();
    }

    private int GetOffset(int x, int y, int channel)
    {
        // YOLO outputs a tensor that has a shape of 125x13x13, which 
        // WinML flattens into a 1D array.  To access a specific channel 
        // for a given (x,y) cell position, we need to calculate an offset
        // into the array
        return (channel * this.channelStride) + (y * COL_COUNT) + x;
    }

    private BoundingBoxDimensions ExtractBoundingBoxDimensions(float[] modelOutput, int x, int y, int channel)
    {
        return new BoundingBoxDimensions
        {
            X = modelOutput[GetOffset(x, y, channel)],
            Y = modelOutput[GetOffset(x, y, channel + 1)],
            Width = modelOutput[GetOffset(x, y, channel + 2)],
            Height = modelOutput[GetOffset(x, y, channel + 3)]
        };
    }

    private float GetConfidence(float[] modelOutput, int x, int y, int channel)
    {
        return Sigmoid(modelOutput[GetOffset(x, y, channel + 4)]);
    }

    private CellDimensions MapBoundingBoxToCell(int x, int y, int box, BoundingBoxDimensions boxDimensions)
    {
        return new CellDimensions
        {
            X = ((float)x + Sigmoid(boxDimensions.X)) * CELL_WIDTH,
            Y = ((float)y + Sigmoid(boxDimensions.Y)) * CELL_HEIGHT,
            Width = (float)Math.Exp(boxDimensions.Width) * CELL_WIDTH * anchors[box * 2],
            Height = (float)Math.Exp(boxDimensions.Height) * CELL_HEIGHT * anchors[box * 2 + 1],
        };
    }

    public float[] ExtractClasses(float[] modelOutput, int x, int y, int channel)
    {
        float[] predictedClasses = new float[CLASS_COUNT];
        int predictedClassOffset = channel + BOX_INFO_FEATURE_COUNT;
        for (int predictedClass = 0; predictedClass < CLASS_COUNT; predictedClass++)
        {
            predictedClasses[predictedClass] = modelOutput[GetOffset(x, y, predictedClass + predictedClassOffset)];
        }
        return Softmax(predictedClasses);
    }

    private ValueTuple<int, float> GetTopResult(float[] predictedClasses)
    {
        return predictedClasses
            .Select((predictedClass, index) => (Index: index, Value: predictedClass))
            .OrderByDescending(result => result.Value)
            .First();
    }

    private float IntersectionOverUnion(RectangleF boundingBoxA, RectangleF boundingBoxB)
    {
        var areaA = boundingBoxA.Width * boundingBoxA.Height;

        if (areaA <= 0)
            return 0;

        var areaB = boundingBoxB.Width * boundingBoxB.Height;

        if (areaB <= 0)
            return 0;

        var minX = Math.Max(boundingBoxA.Left, boundingBoxB.Left);
        var minY = Math.Max(boundingBoxA.Top, boundingBoxB.Top);
        var maxX = Math.Min(boundingBoxA.Right, boundingBoxB.Right);
        var maxY = Math.Min(boundingBoxA.Bottom, boundingBoxB.Bottom);

        var intersectionArea = Math.Max(maxY - minY, 0) * Math.Max(maxX - minX, 0);

        return intersectionArea / (areaA + areaB - intersectionArea);
    }

    /// <summary>
    /// Extract bounding boxes and their probabilities from the prediction output.
    /// </summary>
    //private List<YoloBoundingBox> ExtractBoxes(float[] predictionOutput, float[] anchors)
    //{
    //    //var shape = predictionOutput.Shape;
    //    //Debug.Assert(shape.Count == 4, "The model output has unexpected shape");
    //    //Debug.Assert(shape[0] == 1, "The batch size must be 1");

    //    IReadOnlyList<float> outputs = predictionOutput;

    //    var numAnchor = anchors.Length / 2;
    //    //var channels = shape[1];
    //    //var height = shape[2];
    //    //var width = shape[3];

    //    var channels = CHANNEL_COUNT;
    //    var height = 

    //    Debug.Assert(channels % numAnchor == 0);
    //    var numClass = (channels / numAnchor) - 5;

    //    Debug.Assert(numClass == this.labels.Count);

    //    var boxes = new List<YoloBoundingBox>();
    //    var probs = new List<float[]>();
    //    for (int gridY = 0; gridY < height; gridY++)
    //    {
    //        for (int gridX = 0; gridX < width; gridX++)
    //        {
    //            int offset = 0;
    //            int stride = (int)(height * width);
    //            int baseOffset = gridX + gridY * (int)width;

    //            for (int i = 0; i < numAnchor; i++)
    //            {
    //                var x = (Logistic(outputs[baseOffset + (offset++ * stride)]) + gridX) / width;
    //                var y = (Logistic(outputs[baseOffset + (offset++ * stride)]) + gridY) / height;
    //                var w = (float)Math.Exp(outputs[baseOffset + (offset++ * stride)]) * anchors[i * 2] / width;
    //                var h = (float)Math.Exp(outputs[baseOffset + (offset++ * stride)]) * anchors[i * 2 + 1] / height;

    //                x = x - (w / 2);
    //                y = y - (h / 2);

    //                var objectness = Logistic(outputs[baseOffset + (offset++ * stride)]);

    //                var classProbabilities = new float[numClass];
    //                for (int j = 0; j < numClass; j++)
    //                {
    //                    classProbabilities[j] = outputs[baseOffset + (offset++ * stride)];
    //                }
    //                var max = classProbabilities.Max();
    //                for (int j = 0; j < numClass; j++)
    //                {
    //                    classProbabilities[j] = (float)Math.Exp(classProbabilities[j] - max);
    //                }
    //                var sum = classProbabilities.Sum();
    //                for (int j = 0; j < numClass; j++)
    //                {
    //                    classProbabilities[j] *= objectness / sum;
    //                }

    //                if (classProbabilities.Max() > this.probabilityThreshold)
    //                {
    //                    boxes.Add(new BoundingBox(x, y, w, h));
    //                    probs.Add(classProbabilities);
    //                }
    //            }
    //            Debug.Assert(offset == channels);
    //        }
    //    }

    //    Debug.Assert(boxes.Count == probs.Count);
    //    return new ExtractedBoxes(boxes, probs);
    //}

    public IList<YoloBoundingBox> ParseOutputs(float[] yoloModelOutputs, float threshold = .3F)
    {
        var boxes = new List<YoloBoundingBox>();

        for (int row = 0; row < ROW_COUNT; row++)
        {
            for (int column = 0; column < COL_COUNT; column++)
            {
                for (int box = 0; box < BOXES_PER_CELL; box++)
                {
                    var channel = (box * (CLASS_COUNT + BOX_INFO_FEATURE_COUNT));
                    BoundingBoxDimensions boundingBoxDimensions = ExtractBoundingBoxDimensions(yoloModelOutputs, row, column, channel);
                    float confidence = GetConfidence(yoloModelOutputs, row, column, channel);
                    CellDimensions mappedBoundingBox = MapBoundingBoxToCell(row, column, box, boundingBoxDimensions);
                    if (confidence < threshold)
                        continue;

                    float[] predictedClasses = ExtractClasses(yoloModelOutputs, row, column, channel);
                    var (topResultIndex, topResultScore) = GetTopResult(predictedClasses);
                    var topScore = topResultScore * confidence;

                    if (topScore < threshold)
                        continue;

                    boxes.Add(new YoloBoundingBox()
                    {
                        Dimensions = new BoundingBoxDimensions
                        {
                            X = (mappedBoundingBox.X - mappedBoundingBox.Width / 2),
                            Y = (mappedBoundingBox.Y - mappedBoundingBox.Height / 2),
                            Width = mappedBoundingBox.Width,
                            Height = mappedBoundingBox.Height,
                        },
                        Confidence = topScore,
                        Label = labels[topResultIndex],
                        BoxColor = classColors[topResultIndex]
                    });
                }
            }
        }

        return boxes;
    }

    public IList<YoloBoundingBox> FilterBoundingBoxes(IList<YoloBoundingBox> boxes, int limit, float threshold)
    {
        var activeCount = boxes.Count;
        var isActiveBoxes = new bool[boxes.Count];

        for (int i = 0; i < isActiveBoxes.Length; i++)
            isActiveBoxes[i] = true;

        var sortedBoxes = boxes.Select((b, i) => new { Box = b, Index = i })
                    .OrderByDescending(b => b.Box.Confidence)
                    .ToList();

        var results = new List<YoloBoundingBox>();

        for (int i = 0; i < boxes.Count; i++)
        {
            if (isActiveBoxes[i])
            {
                var boxA = sortedBoxes[i].Box;
                results.Add(boxA);

                if (results.Count >= limit)
                    break;

                for (var j = i + 1; j < boxes.Count; j++)
                {
                    if (isActiveBoxes[j])
                    {
                        var boxB = sortedBoxes[j].Box;

                        if (IntersectionOverUnion(boxA.Rect, boxB.Rect) > threshold)
                        {
                            isActiveBoxes[j] = false;
                            activeCount--;

                            if (activeCount <= 0)
                                break;
                        }
                    }
                }

                if (activeCount <= 0)
                    break;
            }
        }

        return results;
    }

    public ExportedYoloOutputParser()
    {
    }
}

