#include <iostream>
#include "../include/ClassicML/models.h"
#include "../include/ClassicML/macros.h"

Models::Models(Dataset& shareData) : data(shareData), fit(data), error(shareData) {}

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

void LinearRegression::loss()
{

	error.errorsRegression();
}

LinearRegression::~LinearRegression(){}
