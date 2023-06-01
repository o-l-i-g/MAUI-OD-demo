using System;
using Microsoft.ML.Data;

namespace ONNXConsolePort.DataStructures;

public class ImageNetPrediction
{
    [ColumnName("grid")]
    public float[] PredictedLabels;
}

