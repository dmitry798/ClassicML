#include <iostream>
#include "../include/ClassicML/models.h"
#include "../include/ClassicML/macros.h"

Models::Models(Dataset& shareData) : data(shareData), fit(data), error(shareData) {}

LinearRegression::LinearRegression(Dataset& shareData) : Models(shareData) {}

void LinearRegression::trn()
{
	Matrix U;
	Matrix s;
	Matrix VT;

	fit.svd(U, s, VT);

	W = VT.transpose() * s * U.transpose() * Y_train;
}

void LinearRegression::train(int iters, double learning_rate, double partion_save_grade)
{
	fit.nesterov(iters, learning_rate, partion_save_grade);
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
