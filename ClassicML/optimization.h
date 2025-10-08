#pragma once
#include "matrix.h"
#include "preprocessor.h"
using namespace Data;

class Optimizer
{
private:
	Dataset& data;

public:

	Optimizer(Dataset& shareData);

	void stochastic_gradient_descent(double learning_rate, int epoch);

	void nesterov(int iters, double learning_rate, double partion_save_grade);

	void svd(Matrix& U, Matrix& s, Matrix& VT);

	~Optimizer();
};