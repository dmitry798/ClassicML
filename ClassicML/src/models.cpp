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

Matrix Models::predict(Matrix& X_predict) const
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

Matrix LinearRegression::predict() const
{
	StandartScaler scaler(data);
	Y_pred = scaler.denormalize(X_test_norm * W, mean_y, std_y);
	return Y_pred;
}

Matrix LinearRegression::predict(Matrix& X_predict) const
{
    return predict(X_predict);
}

void LinearRegression::loss() const
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

Matrix LogisticRegression::predict() const
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

Matrix LogisticRegression::predict(Matrix& X_predict) const
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

void LogisticRegression::loss(double threshold) const
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

Matrix LogisticRegression::MultiClassLogisticRegression::predict() const
{
    Y_pred = softMax(X_test_norm * W);
    return Y_pred;
}

Matrix LogisticRegression::MultiClassLogisticRegression::predict(Matrix& X_predict) const
{
    Y_pred = softMax(Models::predict(X_predict));
    return Y_pred;
}

void LogisticRegression::MultiClassLogisticRegression::loss(double threshold) const
{
    error.errorsLogClassifier("loglossMulti", threshold);
}

LogisticRegression::MultiClassLogisticRegression::~MultiClassLogisticRegression() {}

/************************************************************************************************************************************/

//методы KNN

Knn::Knn(Dataset& shareData) : Models(shareData)
{

}

void Knn::train()
{

}

Matrix Knn::predict() const
{
    return Matrix();
}

Matrix Knn::predict(Matrix& X_predict) const
{
    return Matrix();
}

void Knn::loss() const
{

}

