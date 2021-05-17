using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet_SpamClassifier
{
    class Program
    {
        static readonly string _path = "..\\..\\..\\Data\\ham-spam.csv";

        static readonly string[] _samples =
        {
            "If you can get the new revenue projections to me by Friday, I'll fold them into the forecast.",
            "Can you attend a meeting in Atlanta on the 16th? I'd like to get the team together to discuss in-person.",
            "Why pay more for expensive meds when you can order them online and save $$$?"
        };

        static void Main(string[] args)
        {
            // 1. Initialize ML.Net Environment
            MLContext mlContext = new MLContext();

            // 2. Load training data
            var data = mlContext.Data.LoadFromTextFile<ModelInput>(_path, hasHeader: true, separatorChar: ',');

            // Split the data into a training set and a test set
            var trainTestData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            // 3. Add data transformation
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: "Text");

            // 4. Add Algorithm
            var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression();
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // 5. Train the model
            Console.WriteLine("Training the model...");
            var model = trainingPipeline.Fit(trainData);

            // 6. Evaluate the model on test data
            var predictions = model.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderPrecisionRecallCurve:P2}");
            Console.WriteLine($"F1: {metrics.F1Score:P2}");

            // 7. Consume Model - Predict on sample data and print the results
            var predictor = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

            foreach (var sample in _samples)
            {
                var input = new ModelInput { Text = sample };
                var prediction = predictor.Predict(input);

                Console.WriteLine();
                Console.WriteLine($"{input.Text}");
                Console.WriteLine($"Spam score: {prediction.Probability}");
                Console.WriteLine($"Classification: {(Convert.ToBoolean(prediction.Prediction) ? "Spam" : "Not spam")}");
            }

            
        }
    }

    public class ModelInput
    {
        [LoadColumn(0), ColumnName("Label")]
        public bool IsSpam;

        [LoadColumn(1)]
        public string Text;
    }

    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Probability { get; set; }
    }
}
