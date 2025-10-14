#include <iostream>
#include "../include/ClassicML/models.h"
#include "../include/ClassicML/macros.h"

Models::Models(Dataset& shareData) : data(shareData), fit(data), error(shareData) {}

Matrix Models::predict(Matrix& X_predict) const
{
    StandartScaler scaler(data);

    Matrix mean(X_predict.getCols(), 1, "mean"); Matrix std(X_predict.getCols(), 1, "std");
    mean.mean(X_predict); std.std(X_predict, mean);

    Matrix&& X_predict_norm = scaler.normalize(X_predict, mean, std);

    return X_predict_norm * W;
}

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
    else if (method == "nesterov") fit.sgdNesterov(iters, lr, mini_batch, gamma, 1);
    else if (method == "sgd") fit.sgd(iters, lr, mini_batch, 1);
    else if (method == "gd") fit.gradientDescent(iters, lr, 1);
    else throw std::runtime_error("Unknown training method: " + method);
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


//методы логистической регрессии

LogisticRegression::LogisticRegression(Dataset& shareData, string way) : Models(shareData), way(way), model(shareData) {}

void LogisticRegression::train(const string& method, int iters, double lr, int mini_batch, double gamma)
{
    if (way == "binary")
    {
        if (method == "nesterov") fit.sgdNesterov(iters, lr, mini_batch, gamma, 2);
        else if (method == "sgd") fit.sgd(iters, lr, mini_batch, 2);
        else if (method == "gd") fit.gradientDescent(iters, lr, 2);
        else throw std::runtime_error("Unknown training method: " + method);
    }
    else if (way == "multi")
    {
        model.train(method, iters, lr, mini_batch, gamma);
    }
}

Matrix LogisticRegression::predict() const
{
    if (way == "binary")
        return sigmoid(X_test_norm * W);
    else if (way == "multi")
        return model.predict();
}

Matrix LogisticRegression::predict(Matrix& X_predict) const
{
    if (way == "binary")
        return sigmoid(Models::predict(X_predict));
    else if (way == "multi")
        return model.predict(X_predict);
}

void LogisticRegression::loss() const
{
    if (way == "binary")
        cout << error.logLoss();
    else if (way == "multi")
        model.loss();
}

LogisticRegression::~LogisticRegression() {}


//методы мультиклассовой логистической регрессии

LogisticRegression::MultiClassLogisticRegression::MultiClassLogisticRegression(Dataset& shareData) : Models(shareData) {}

void LogisticRegression::MultiClassLogisticRegression::train(const string& method, int iters, double lr, int mini_batch, double gamma)
{
    if (method == "nesterov") fit.sgdNesterov(iters, lr, mini_batch, gamma, 3);
    else if (method == "sgd") fit.sgd(iters, lr, mini_batch, 3);
    else if (method == "gd") fit.gradientDescent(iters, lr, 3);
    else throw std::runtime_error("Unknown training method: " + method);
}

Matrix LogisticRegression::MultiClassLogisticRegression::predict() const
{
    Matrix res;
    double sum = 0.0;

    for (int i = 0; i < X_test_norm.getRows(); i++)
    {
        softMax(X_test_norm.sliceRow(i, i + 1) * W).print();
    }
    return res;
}

Matrix LogisticRegression::MultiClassLogisticRegression::predict(Matrix& X_predict) const
{
    return softMax(Models::predict(X_predict));
}

void LogisticRegression::MultiClassLogisticRegression::loss() const
{
    cout << error.logLossM();
}

LogisticRegression::MultiClassLogisticRegression::~MultiClassLogisticRegression() {}