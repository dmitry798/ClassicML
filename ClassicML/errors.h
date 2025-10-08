#pragma once
#include "matrix.h"
#include "preprocessor.h"
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

	double Accuracy();

	double Precision();

	double Recall();

	double F1Score();

	double RocAuc();
};

