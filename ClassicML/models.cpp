#include <iostream>
#include "models.h"
#include "macros.h"

Models::Models(Dataset& shareData) : data(shareData), fit(data), error(shareData) {}

LinearRegression::LinearRegression(Dataset& shareData) : Models(shareData) {}

void LinearRegression::train()
{
	Matrix U;
	Matrix s;
	Matrix VT;

	fit.svd(U, s, VT);

	W = VT.transpose() * s * U.transpose() * Y_train;
}

Matrix LinearRegression::predict() const
{
	return X_test * W;
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
