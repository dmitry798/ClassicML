#pragma once
#include "matrix.h"
#include "preprocessor.h"
#include "optimization.h"
using namespace Data;


class Errors
{
private:
	Dataset& data;
public:
	Errors(Dataset& data);

	double MSE() const;

	double RMSE() const;

	double MAE() const;

	double R2() const;

	void errorsRegression() const;

	double logLoss() const;

	double accuracy();

	double precision();

	double recall();

	double f1Score();

	double rocAuc();
};

