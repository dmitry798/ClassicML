#pragma once
#include "matrix_operations.h"
#include "preprocessor.h"

class Fit
{
private:
	Data& data;

	Matrix W;

	//для СВД
	Matrix U;
	Matrix s;
	Matrix VT;
public:

	/*Fit();*/

	Fit(Data& shareData);

	/*void stochastic_gradient_descent(double learning_rate, int epoch);

	void nesterov();*/

	Matrix svd();

	/*~Fit();*/
};