#include <iostream>
#include "../include/ClassicML/models.h"
#include "../include/ClassicML/macros.h"

/************************************************************************************************************************************/

//выбор оптимизаторов

Models::Trainer::Trainer() {}

void Models::Trainer::choice_train(const string& method, Optimizer& fit, int iters, double lr, int mini_batch, double gamma, int num_method)
{
    if (method == "nesterov") fit.sgdNesterov(iters, lr, mini_batch, gamma, num_method);
    else if (method == "sgd") fit.sgd(iters, lr, mini_batch, num_method);
    else if (method == "gd") fit.gradientDescent(iters, lr, num_method);
    else if (method == "momentum") fit.sgdMomentum(iters, lr, mini_batch, gamma, num_method);
    else throw std::runtime_error("Unknown training method: " + method);
}

/************************************************************************************************************************************/

//общая структура модели

Models::Models(Dataset& shareData) : data(shareData), fit(data), error(shareData) {}

Matrix Models::predict(Matrix& X_predict)
{
    StandartScaler scaler(data);

    Matrix mean(X_predict.getCols(), 1, "mean"); Matrix std(X_predict.getCols(), 1, "std");
    mean.mean(X_predict); std.std(X_predict, mean);

    Matrix&& X_predict_norm = scaler.normalize(X_predict, mean, std);

    return X_predict_norm * W;
}

/************************************************************************************************************************************/

//методы линейной регрессии
LinearRegression::LinearRegression(Dataset& shareData) : Models(shareData) {}

void LinearRegression::train(const string& method, int iters, double lr, int mini_batch, double gamma)
{
    if (method == "svd")
    {
        Matrix U, s, VT;
        fit.svd(U, s, VT);
        W = VT.transpose() * s * U.transpose() * Y_train;
    }
    else trainer.choice_train(method, fit, iters, lr, mini_batch, gamma);
}

Matrix LinearRegression::predict()
{
	StandartScaler scaler(data);
	Y_pred = scaler.denormalize(X_test_norm * W, mean_y, std_y);
	return Y_pred;
}

Matrix LinearRegression::predict(Matrix& X_predict)
{
    return Models::predict(X_predict);
}

void LinearRegression::loss()
{
	error.errorsRegression();
}

LinearRegression::~LinearRegression(){}

/************************************************************************************************************************************/

//методы логистической регрессии

LogisticRegression::LogisticRegression(Dataset& shareData, string way) : Models(shareData), way(way), model(shareData) {}

void LogisticRegression::train(const string& method, int iters, double lr, int mini_batch, double gamma)
{
    if (way == "binary")
    {
        trainer.choice_train(method, fit, iters, lr, mini_batch, gamma);
    }
    else if (way == "multi")
    {
        model.train(method, iters, lr, mini_batch, gamma);
    }
}

Matrix LogisticRegression::predict()
{
    if (way == "binary")
    {
        Y_pred = sigmoid(X_test_norm * W);
        return Y_pred;
    }
    else if (way == "multi")
    {
        return model.predict();
    }
}

Matrix LogisticRegression::predict(Matrix& X_predict)
{
    if (way == "binary")
    {
        Y_pred = sigmoid(Models::predict(X_predict));
        return Y_pred;
    }
    else if (way == "multi")
    {
        return model.predict(X_predict);
    }
}

void LogisticRegression::loss(double threshold)
{
    if (way == "binary")
        error.errorsLogClassifier("logloss", threshold);
    else if (way == "multi")
        model.loss(threshold);
}

LogisticRegression::~LogisticRegression() {}

/************************************************************************************************************************************/

//методы мультиклассовой логистической регрессии

LogisticRegression::MultiClassLogisticRegression::MultiClassLogisticRegression(Dataset& shareData) : Models(shareData) {}

void LogisticRegression::MultiClassLogisticRegression::train(const string& method, int iters, double lr, int mini_batch, double gamma)
{
    trainer.choice_train(method, fit, iters, lr, mini_batch, gamma);
}

Matrix LogisticRegression::MultiClassLogisticRegression::predict()
{
    Y_pred = softMax(X_test_norm * W);
    return Y_pred;
}

Matrix LogisticRegression::MultiClassLogisticRegression::predict(Matrix& X_predict)
{
    Y_pred = softMax(Models::predict(X_predict));
    return Y_pred;
}

void LogisticRegression::MultiClassLogisticRegression::loss(double threshold)
{
    error.errorsLogClassifier("loglossMulti", threshold);
}

LogisticRegression::MultiClassLogisticRegression::~MultiClassLogisticRegression() {}

/************************************************************************************************************************************/

// Методы К ближайших соседей для классификации

Knn::Knn(Dataset& shareData, int num_neighbors) : Models(shareData), num_neighbors(num_neighbors), dist_method(shareData) {}

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
                if (Y_train(i, j) == 1)
                    concate(i, 1) = j + 1;
        }
        //сортируем по расстоянию
        for (int i = 0; i < concate.getRows() - 1; i++) 
        {
            for (int j = 0; j < concate.getRows() - i - 1; j++) 
            {
                if (concate(j, 0) > concate(j + 1, 0)) 
                {
                    for (int k = 0; k < concate.getCols(); k++) 
                    {
                        double temp = concate(j, k);
                        concate(j, k) = concate(j + 1, k);
                        concate(j + 1, k) = temp;
                    }
                }
            }
        }

        Matrix sorted = concate.sliceRow(0, num_neighbors);

        //выбираем часто встречающиеся
        int pred = mode(sorted.sliceCols(1, 2));
        Y_pred(t, 0) = pred;
    }
    return Y_pred;
}

Matrix Knn::predict(Matrix& X_predict, string distance)
{
    StandartScaler scaler(data);

    Matrix mean(X_predict.getCols(), 1, "mean"); Matrix std(X_predict.getCols(), 1, "std");
    mean.mean(X_predict); std.std(X_predict, mean);

    Matrix&& X_predict_norm = scaler.normalize(X_predict, mean, std);


    Y_pred = Matrix(X_predict_norm.getRows(), 1);
    //проходимся по всем тестовым точкам
    for (int t = 0; t < X_predict_norm.getRows(); t++)
    {
        Matrix dist;

        //рассчитываем расстояние
        if (distance == "evklid") dist = dist_method.evklid(move(X_predict_norm.sliceRow(t, t + 1)));
        else if (distance == "manhattan") dist = dist_method.manhattan(move(X_predict_norm.sliceRow(t, t + 1)));

        //делаем матрицу из расстояний и принадлежности к классу
        Matrix concate(X_train_norm.getRows(), 2, "concate-X+Y");
        for (int i = 0; i < X_train_norm.getRows(); i++)
        {
            concate(i, 0) = dist[i];
            for (int j = 0; j < Y_train.getCols(); j++)
                if (Y_train(i, j) == 1)
                    concate(i, 1) = j + 1;
        }
        //сортируем по расстоянию
        for (int i = 0; i < concate.getRows() - 1; i++) 
        {
            for (int j = 0; j < concate.getRows() - i - 1; j++) 
            {
                if (concate(j, 0) > concate(j + 1, 0)) 
                {
                    for (int k = 0; k < concate.getCols(); k++) 
                    {
                        double temp = concate(j, k);
                        concate(j, k) = concate(j + 1, k);
                        concate(j + 1, k) = temp;
                    }
                }
            }
        }

        Matrix sorted = concate.sliceRow(0, num_neighbors);

        //выбираем часто встречающиеся
        int pred = mode(sorted.sliceCols(1, 2));
        Y_pred(t, 0) = pred;
    }
    return Y_pred;
}

void Knn::loss(double threshold)
{
    error.errorsLogClassifier("loglossMulti", threshold);
}

Knn::~Knn() {}

/************************************************************************************************************************************/

//методы К ближайших соседей для регрессии

KnnRegression::KnnRegression(Dataset& shareData, int num_neighbors) : Models(shareData), num_neighbors(num_neighbors), dist_method(shareData) {}

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
            for(int j = 0; j < Y_train.getCols(); j++)
                concate(i, j + 1) = Y_train(i, j);
        }

        //сортируем по расстоянию
        for (int i = 0; i < concate.getRows() - 1; i++) 
        {
            for (int j = 0; j < concate.getRows() - i - 1; j++) 
            {
                if (concate(j, 0) > concate(j + 1, 0)) 
                {
                    for (int k = 0; k < concate.getCols(); k++) 
                    {
                        double temp = concate(j, k);
                        concate(j, k) = concate(j + 1, k);
                        concate(j + 1, k) = temp;
                    }
                }
            }
        }

        Matrix sorted = concate.sliceRow(0, num_neighbors);

        //находим среднее
        for (int j = 0; j < Y_train.getCols(); j++) 
        {
            double sum = 0.0;
            for (int i = 0; i < num_neighbors; i++)
                sum += sorted(i, j + 1);
            Y_pred(t, j) = sum / num_neighbors;
        }
    }
    return Y_pred;
}

Matrix KnnRegression::predict(Matrix& X_predict, string distance)
{
    StandartScaler scaler(data);

    Matrix mean(X_predict.getCols(), 1, "mean"); Matrix std(X_predict.getCols(), 1, "std");
    mean.mean(X_predict); std.std(X_predict, mean);

    Matrix&& X_predict_norm = scaler.normalize(X_predict, mean, std);


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
        for (int i = 0; i < concate.getRows() - 1; i++)
        {
            for (int j = 0; j < concate.getRows() - i - 1; j++)
            {
                if (concate(j, 0) > concate(j + 1, 0))
                {
                    for (int k = 0; k < concate.getCols(); k++)
                    {
                        double temp = concate(j, k);
                        concate(j, k) = concate(j + 1, k);
                        concate(j + 1, k) = temp;
                    }
                }
            }
        }

        Matrix sorted = concate.sliceRow(0, num_neighbors);

        //находим среднее
        for (int j = 0; j < Y_train.getCols(); j++)
        {
            double sum = 0.0;
            for (int i = 0; i < num_neighbors; i++)
                sum += sorted(i, j + 1);
            Y_pred(t, j) = sum / num_neighbors;
        }
    }
    return Y_pred;
}

void KnnRegression::loss()
{
    error.errorsRegression();
}

KnnRegression::~KnnRegression() {}