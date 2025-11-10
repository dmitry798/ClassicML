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

	double accuracy(double threshold = 0.5) const;

	double accuracyMultiClss() const;

	double precision(double threshold = 0.5) const;

	double recall(double threshold = 0.5) const;

	double f1Score(double threshold = 0.5) const;

	double inertia(Matrix centroids) const;

	void errorsLogClassifier(string name, double threshold = 0.5) const;

	void errorsKnnClassifier(double threshold = 0.5) const;
};

