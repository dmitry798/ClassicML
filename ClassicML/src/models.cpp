#include <iostream>
#include "../include/ClassicML/models.h"
#include "../include/ClassicML/macros.h"

Models::Models(Dataset& shareData) : data(shareData), fit(data), error(shareData) {}


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
    else if (method == "nesterov") 
    {
        fit.sgdNesterov(iters, lr, mini_batch, gamma);
    }
    else if (method == "sgd")
    {
        fit.sgd(iters, lr, mini_batch);
    }
    else 
    {
        throw std::runtime_error("Unknown training method: " + method);
    }
}

Matrix LinearRegression::predict() const
{
	StandartScaler scaler(data);
	Y_pred = scaler.denormalize(X_test_norm * W, mean_y, std_y);
	return Y_pred;
}

Matrix LinearRegression::predict(Matrix& X_predict) const
{
	return X_predict * W;
}

void LinearRegression::loss() const
{

	error.errorsRegression();
}

LinearRegression::~LinearRegression(){}


//методы логистической регрессии

LogisticRegression::LogisticRegression(Dataset& shareData) : Models(shareData) {}

void LogisticRegression::train(const string& method, int iters, double lr, int mini_batch, double gamma)
{
    if (method == "nesterov")
    {
        fit.sgdNesterov(iters, lr, mini_batch, gamma, true);
    }
    else if (method == "sgd")
    {
        fit.sgd(iters, lr, mini_batch, true);
    }
    else
    {
        throw std::runtime_error("Unknown training method: " + method);
    }
}

Matrix LogisticRegression::predict() const
{
    return sigmoid(X_test_norm * W);
}

Matrix LogisticRegression::predict(Matrix& X_predict) const
{
    return sigmoid(X_predict * W);
}

void LogisticRegression::loss() const
{

    cout << error.logLoss();
}

LogisticRegression::~LogisticRegression() {}