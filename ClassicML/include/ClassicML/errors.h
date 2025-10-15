#pragma once
#include "matrix.h"
#include "preprocessor.h"
#include "optimization.h"


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

	double logLossMulti() const;

	double accuracy(double threshold) const;

	double precision(double threshold) const;

	double recall(double threshold) const;

	double f1Score(double threshold) const;

	double rocAuc() const;

	void errorsLogClassifier(string name, double threshold) const;
};

