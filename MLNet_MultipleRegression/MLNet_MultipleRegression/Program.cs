using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;

namespace MLNet_MultipleRegression
{
    class Program
    {
        static readonly string _path = "..\\..\\..\\Data\\pacific-heights.csv";

        static void Main(string[] args)
        {
            // 1. Initialize ML.Net Environment
            MLContext mlContext = new MLContext();

            // 2. Load training Data
            IDataView data = mlContext.Data.LoadFromTextFile<ModelInput>(_path, hasHeader: true, separatorChar: ',');

            // 3. Split the data into a training set and test set
            var trainTestData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            // 4. Add data transformations
            // One-hot encode the string values in the "UseCode" column and train the model 
            var dataProcessPipeline = mlContext.Transforms.Categorical.OneHotEncoding(inputColumnName: "UseCode", outputColumnName: "UseCodeEncoded")
                .Append(mlContext.Transforms.Concatenate(outputColumnName: "Features", "UseCodeEncoded", "Bathrooms", "Bedrooms", "TotalRooms", "FinishedSquareFeet"));

            // 5. Add algorithm
            var trainer = mlContext.Regression.Trainers.FastForest(numberOfTrees: 200, minimumExampleCountPerLeaf: 4);
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // 6. Train the model
            var model = trainingPipeline.Fit(trainData);

            // 7. Evaluate model on test data
            var predictions = model.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(predictions);
            Console.WriteLine($"R2 score: {metrics.RSquared:0.###}");

            // 8. Evaluate the model again using cross-validation
            var scores = mlContext.Regression.CrossValidate(data, trainingPipeline, numberOfFolds: 5);
            var mean = scores.Average(x => x.Metrics.RSquared);
            Console.WriteLine($"Mean cross-validated R2 score: {mean:0.##}");

            //9. Model Consumption
            // Predict on sample data and print results

            var input = new ModelInput
            {
                Bathrooms = 1.0f,
                Bedrooms = 1.0f,
                TotalRooms = 3.0f,
                FinishedSquareFeet = 653.0f,
                UseCode = "Condominium",
                LastSoldPrice = 0.0f
            };

            var predictor_result = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model).Predict(input);
            Console.WriteLine($"Predicted price: ${predictor_result.Price:n0}; Actual price: $665,000");


        }
    }

    public class ModelInput
    {
        [LoadColumn(1)]
        public float Bathrooms;

        [LoadColumn(2)]
        public float Bedrooms;

        [LoadColumn(3)]
        public float FinishedSquareFeet;

        [LoadColumn(5), ColumnName("Label")]
        public float LastSoldPrice;

        [LoadColumn(9)]
        public float TotalRooms;

        [LoadColumn(10)]
        public string UseCode;

    }

    public class ModelOutput 
    {
        [ColumnName("Score")]
        public float Price;
    }
}
