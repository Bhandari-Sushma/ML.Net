using Microsoft.ML;
using Microsoft.ML.Data;
using System;


namespace MLNet_SimpleRegression
{
    class Program
    {
        static readonly string _path = "..\\..\\..\\Data\\poverty.csv";
        static void Main(string[] args)
        {
            // 1. Initialize the ML.Net Environment
            MLContext mlContext = new MLContext();

            // 2. Load training data
            IDataView data = mlContext.Data.LoadFromTextFile<ModelInput>(_path, hasHeader: true, separatorChar:',');

            // 3. Add data transformation
            var dataProcessPipeline = mlContext.Transforms.NormalizeMinMax("PovertyRate")
                .Append(mlContext.Transforms.Concatenate(outputColumnName:"Features", inputColumnNames:"PovertyRate"));


            // 4. Add algorithm
            var trainer = mlContext.Regression.Trainers.Sdca(featureColumnName: "Features", labelColumnName: "BirthRate");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // 5. Train the model
            var model = trainingPipeline.Fit(data);

            // 6. Predict on sample data and print the results
            var input = new ModelInput { PovertyRate = 19.7f };
            var predictor = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);
            var prediction_result = predictor.Predict(input);

            Console.WriteLine($"Predicted birth rate:{prediction_result.BirthRate}");




        }
    }

    public class ModelInput
    {
        [LoadColumn(1)]
        public float PovertyRate;
        [LoadColumn(5)]
        public float BirthRate;

    }

    public class ModelOutput
    {
        [ColumnName("Score")]
        public float BirthRate;
    }
}
