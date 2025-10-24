#include "../include/ClassicML/models.h"
#include "../include/ClassicML/macros.h"

/************************************************************************************************************************************/

//методы К ближайших соседей для регрессии

KnnRegression::KnnRegression(Dataset& shareData, int num_neighbors, string weighted) : Models(shareData), num_neighbors(num_neighbors), dist_method(shareData), weighted(weighted) {}

Matrix KnnRegression::predict(string distance)
{
    Y_pred = Matrix(X_test_norm.getRows(), Y_train.getCols());
    //проходимся по всем тестовым точкам
    for (int t = 0; t < X_test_norm.getRows(); t++)
    {
        Matrix dist;

        //рассчитываем расстояние
        if (distance == "evklid") dist = dist_method.evklid(move(X_test_norm.sliceRow(t, t + 1)));
        else if (distance == "manhattan") dist = dist_method.manhattan(move(X_test_norm.sliceRow(t, t + 1)));

        //делаем матрицу из расстояний и значений
        Matrix concate(X_train_norm.getRows(), 1 + Y_train.getCols(), "concate-X+Y");
        for (int i = 0; i < X_train_norm.getRows(); i++)
        {
            concate(i, 0) = dist[i];
            for (int j = 0; j < Y_train.getCols(); j++)
                concate(i, j + 1) = Y_train(i, j);
        }

        //сортируем по расстоянию
        concate.sortRows(0);

        Matrix sorted = concate.sliceRow(0, num_neighbors);

        //находим среднее
        if (weighted == "uniform")
        {
            for (int j = 0; j < Y_train.getCols(); j++)
            {
                double sum = 0.0;
                for (int i = 0; i < num_neighbors; i++)
                    sum += sorted(i, j + 1);
                Y_pred(t, j) = sum / num_neighbors;
            }
        }
        //наиболее весомые
        else if (weighted == "distance")
        {
            Matrix distances = sorted.sliceCols(0, 1);

            for (int j = 0; j < Y_train.getCols(); j++)
            {
                double weighted_sum = 0.0;
                double weight_total = 0.0;
                Matrix w = 1 / distances;

                for (int i = 0; i < distances.getRows(); i++)
                {

                    weighted_sum += sorted(i, j + 1) * w[i];
                    weight_total += w[i];
                }

                Y_pred(t, j) = weighted_sum / weight_total;
            }
        }
    }
    return Y_pred;
}

Matrix KnnRegression::predict(Matrix& X_predict, string distance)
{
    StandardScaler scaler(data);

    Matrix mean_(X_predict.getCols(), 1, "mean"); Matrix std_(X_predict.getCols(), 1, "std");
    mean_ = mean(X_predict); std_ = stddev(X_predict, mean_);

    Matrix&& X_predict_norm = scaler.normalize(X_predict, mean_, std_);


    Y_pred = Matrix(X_predict_norm.getRows(), Y_train.getCols());
    //проходимся по всем тестовым точкам
    for (int t = 0; t < X_predict_norm.getRows(); t++)
    {
        Matrix dist;

        //рассчитываем расстояние
        if (distance == "evklid") dist = dist_method.evklid(move(X_predict_norm.sliceRow(t, t + 1)));
        else if (distance == "manhattan") dist = dist_method.manhattan(move(X_predict_norm.sliceRow(t, t + 1)));

        //делаем матрицу из расстояний и принадлежности к классу
        Matrix concate(X_train_norm.getRows(), 1 + Y_train.getCols(), "concate-X+Y");
        for (int i = 0; i < X_train_norm.getRows(); i++)
        {
            concate(i, 0) = dist[i];
            for (int j = 0; j < Y_train.getCols(); j++)
                concate(i, j + 1) = Y_train(i, j);
        }
        //сортируем по расстоянию
        concate.sortRows(0);

        Matrix sorted = concate.sliceRow(0, num_neighbors);

        //находим среднее
        if (weighted == "uniform")
        {
            for (int j = 0; j < Y_train.getCols(); j++)
            {
                double sum = 0.0;
                for (int i = 0; i < num_neighbors; i++)
                    sum += sorted(i, j + 1);
                Y_pred(t, j) = sum / num_neighbors;
            }
        }
        //наиболее весомые
        else if (weighted == "distance")
        {
            Matrix distances = sorted.sliceCols(0, 1);

            for (int j = 0; j < Y_train.getCols(); j++)
            {
                double weighted_sum = 0.0;
                double weight_total = 0.0;
                Matrix w = 1 / distances;

                for (int i = 0; i < distances.getRows(); i++)
                {
                    weighted_sum += sorted(i, j + 1) * w[i];
                    weight_total += w[i];
                }

                Y_pred(t, j) = weighted_sum / weight_total;
            }
        }
    }
    return Y_pred;
}

void KnnRegression::loss()
{
    error.errorsRegression();
}

KnnRegression::~KnnRegression() {}

/************************************************************************************************************************************/