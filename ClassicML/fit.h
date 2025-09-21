#pragma once
#include "matrix_operations.h"

class Fit
{
private:
	Matrix X;
	Matrix W;
	Matrix Y;
public:

	Fit();

	void gradient_descent(double learning_rate, int epoch);

	void stochastic_gradient_descent();

	void nesterov();

	void SVD();

	~Fit();
};