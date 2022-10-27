using Microsoft.ML.Data;

namespace MovieRecommender
{
    //Especificando classe de dados de entrada.
    //LoadColumn especifica quais colunas no conjuto de dados devem ser carregadas.
    public class MovieRating
    {
    [LoadColumn(0)]
    public float userId;
    [LoadColumn(1)]
    public float movieId;
    [LoadColumn(2)]
    public float Label;
    }

    //Classe criada para representar os resultados previtos.
    public class MovieRatingPrediction
    {
    public float Label;
    public float Score;
    }
}