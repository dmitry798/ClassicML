#include "../include/ClassicML/models.h"
#include "../include/ClassicML/macros.h"

/************************************************************************************************************************************/

// Методы К ближайших соседей для классификации

Knn::Knn(Dataset& shareData, int num_neighbors, string weighted) : Models(shareData), num_neighbors(num_neighbors), dist_method(shareData), weighted(weighted) {}

Matrix Knn::predict(string distance)
{
    Y_pred = Matrix(X_test_norm.getRows(), 1);
    //проходимся по всем тестовым точкам
    for (int t = 0; t < X_test_norm.getRows(); t++)
    {
        Matrix dist;
        //рассчитываем расстояние
        if (distance == "evklid") dist = dist_method.evklid(move(X_test_norm.sliceRow(t, t + 1)));
        else if (distance == "manhattan") dist = dist_method.manhattan(move(X_test_norm.sliceRow(t, t + 1)));

        //делаем матрицу из расстояний и принадлежности к классу
        Matrix concate(X_train_norm.getRows(), 2, "concate-X+Y");
        for (int i = 0; i < X_train_norm.getRows(); i++)
        {
            concate(i, 0) = dist[i];
            for (int j = 0; j < Y_train.getCols(); j++)
            {
                if (Y_train(i, j) == 1)
                {
                    concate(i, 1) = j;
                }
            }
        }
        //сортируем по расстоянию
        concate.sortRows(0);

        Matrix sorted = concate.sliceRow(0, num_neighbors);

        //выбираем часто встречающиеся
        int pred = 0;
        if (weighted == "uniform")
        {
            pred = mode(sorted.sliceCols(1, 2));
        }
        //наиболее весомые
        else if (weighted == "distance")
        {
            Matrix distances = sorted.sliceCols(0, 1);
            Matrix classes = sorted.sliceCols(1, 2);

            Matrix weights = 1 / distances;

            Matrix unique_classes = classes.unique();
            double max_weight_sum = -1;
            int best_class = -1;

            for (int u = 0; u < unique_classes.getRows(); u++)
            {
                double cls = unique_classes[u];
                double weight_sum = 0.0;

                for (int i = 0; i < classes.getRows(); i++)
                {
                    if (classes[i] == cls)
                    {
                        weight_sum += weights[i];
                    }
                }

                if (weight_sum > max_weight_sum)
                {
                    max_weight_sum = weight_sum;
                    best_class = int(cls);
                }
            }

            pred = best_class;
        }
        Y_pred[t] = pred;
    }
    return Y_pred;
}

Matrix Knn::predict(Matrix& X_predict, string distance)
{
    StandardScaler scaler(data);

    Matrix mean_(X_predict.getCols(), 1, "mean"); Matrix std_(X_predict.getCols(), 1, "std");
    mean_ = mean(X_predict); std_ = stddev(X_predict, mean_);

    Matrix&& X_predict_norm = scaler.normalize(X_predict, mean_, std_);


    Y_pred = Matrix(X_predict_norm.getRows(), 1);
    //проходимся по всем тестовым точкам
    for (int t = 0; t < X_predict_norm.getRows(); t++)
    {
        Matrix dist;

        //рассчитываем расстояние
        if (distance == "evklid") dist = dist_method.evklid(move(X_predict_norm.sliceRow(t, t + 1)));
        else if (distance == "manhattan") dist = dist_method.manhattan(move(X_predict_norm.sliceRow(t, t + 1)));

        //делаем матрицу из расстояний и принадлежности к классу
        Matrix concate(X_predict_norm.getRows(), 2, "concate-X+Y");
        for (int i = 0; i < X_predict_norm.getRows(); i++)
        {
            concate(i, 0) = dist[i];
            for (int j = 0; j < Y_train.getCols(); j++)
                if (Y_train(i, j) == 1)
                    concate(i, 1) = j;
        }
        //сортируем по расстоянию
        concate.sortRows(0);

        Matrix sorted = concate.sliceRow(0, num_neighbors);

        //выбираем часто встречающиеся
        int pred = 0;
        if (weighted == "uniform")
            pred = mode(sorted.sliceCols(1, 2));
        //наиболее весомые
        else if (weighted == "distance")
        {
            Matrix distances = sorted.sliceCols(0, 1);
            Matrix classes = sorted.sliceCols(1, 2);

            Matrix weights = 1 / distances;

            Matrix unique_classes = classes.unique();
            double max_weight_sum = -1;
            int best_class = -1;

            for (int u = 0; u < unique_classes.getRows(); u++)
            {
                double cls = unique_classes[u];
                double weight_sum = 0.0;

                for (int i = 0; i < classes.getRows(); i++)
                {
                    if (classes[i] == cls)
                        weight_sum += weights[i];
                }

                if (weight_sum > max_weight_sum)
                {
                    max_weight_sum = weight_sum;
                    best_class = int(cls);
                }
            }

            pred = best_class;
        }
        Y_pred[t] = pred;
    }
    return Y_pred;
}

void Knn::loss(double threshold = 0.5)
{
    Y_test = DecoderOHE(Y_test);
    error.errorsKnnClassifier(threshold);
}

Knn::~Knn() {}

/************************************************************************************************************************************/