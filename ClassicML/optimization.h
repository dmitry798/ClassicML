#pragma once
#include "matrix.h"
#include "preprocessor.h"
using namespace Data;

class Fit
{
private:
	Dataset& data;

public:

	Fit(Dataset& shareData);

	void stochastic_gradient_descent(double learning_rate, int epoch);

	void nesterov();

	void svd(Matrix& U, Matrix& s, Matrix& VT);

	~Fit();
};