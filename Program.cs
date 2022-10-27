using Microsoft.ML;
using Microsoft.ML.Trainers;
using MovieRecommender;


//Ponto de partida para as operacoes ML.NET
/*A inicialização mlContextcria um novo ambiente
ML.NET que pode ser compartilhado entre os objetos de fluxo
de trabalho de criação de modelo*/
MLContext mlContext = new MLContext();

//IDataView descreve dados tabulares(numericos e textos)
(IDataView training, IDataView test) LoadData(MLContext mlContext)
{
    //Variaveis training com o caminho da pasta Data com os arquivos .csv
    var trainingDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-train.csv");
    var testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-test.csv");

    IDataView trainingDataView = mlContext.Data.LoadFromTextFile<MovieRating>(trainingDataPath, hasHeader: true, separatorChar: ',');
    IDataView testDataView = mlContext.Data.LoadFromTextFile<MovieRating>(testDataPath, hasHeader: true, separatorChar: ',');

    //Retorna um arquivo IDataView
    return (trainingDataView, testDataView);
}

//Chamando o metoto e retornando os dados de train e test.
(IDataView trainingDataView, IDataView testDataView) = LoadData(mlContext);

//Chamando o metodo e retornando modelo treinado
ITransformer model = BuildAndTrainModel(mlContext, trainingDataView);

ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainingDataView)
{   
    /*Transformando userId e movieId em uma coluna de tipo chave numerica Feature
    que é um formato aceito por algoritmos de recomendacao*/
    IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "userId")
    .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "movieIdEncoded", inputColumnName: "movieId"));
    
    //Algoritmo de treinamento de recomendacao(fatoracao de matriz)
    var options = new MatrixFactorizationTrainer.Options
{
    MatrixColumnIndexColumnName = "userIdEncoded",
    MatrixRowIndexColumnName = "movieIdEncoded",
    LabelColumnName = "Label",
    NumberOfIterations = 20,
    ApproximationRank = 100
};

    var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

    //Treinando modelo
    Console.WriteLine("=============== Treinando o modelo ===============");
    ITransformer model = trainerEstimator.Fit(trainingDataView);

    return model;
}

//Chamando o metodo para avaliar o modelo
EvaluateModel(mlContext, testDataView, model);

void EvaluateModel(MLContext mlContext, IDataView testDataView, ITransformer model)
{
    Console.WriteLine("=============== Avaliando o modelo ===============");

    /*O método Transform() faz previsões para várias linhas de entrada
    fornecidas de um conjunto de dados de teste.*/
    var prediction = model.Transform(testDataView);

    //Avaliando o modelo
    var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");

    //Imprimir metricas de avaliacao
    Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString());
    Console.WriteLine("RSquared: " + metrics.RSquared.ToString());
}

UseModelForSinglePrediction(mlContext, model);


void UseModelForSinglePrediction(MLContext mlContext, ITransformer model)
{
    Console.WriteLine("=============== Realizando uma previsão ===============");
    var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);

    var testInput = new MovieRating { userId = 11, movieId = 200 };

    var movieRatingPrediction = predictionEngine.Predict(testInput);

    if (Math.Round(movieRatingPrediction.Score, 1) > 3.5)
    {
        Console.WriteLine(" O filme " + testInput.movieId + " é recomendado para esse usuário " + testInput.userId);
    }
    else
    {
        Console.WriteLine("O filme " + testInput.movieId + " não é recomendado para esse usuário " + testInput.userId);
    }

}

SaveModel(mlContext, trainingDataView.Schema, model);

void SaveModel(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
{
    var modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "MovieRecommenderModel.zip");

    Console.WriteLine("=============== Salvando o modelo em um arquivo ===============");
    mlContext.Model.Save(model, trainingDataViewSchema, modelPath);
}