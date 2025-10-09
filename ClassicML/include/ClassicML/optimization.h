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

	void sgd(int iters = 1000, double learning_rate = 0.01, int mini_batch = 8);

	void sgdNesterov(int iters = 1000, double learning_rate = 0.01, int mini_batch = 8, double partion_save_grade = 0.01);

	void svd(Matrix& U, Matrix& s, Matrix& VT);

	~Optimizer();
};