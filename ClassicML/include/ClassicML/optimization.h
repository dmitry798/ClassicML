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

	//Стохастический градиетный спуск (mini batch)
	void sgd(int iters = 1000, double learning_rate = 0.01, int mini_batch = 8);

	//Стохастический градиетный спуск методом Нестерова (mini batch)
	void sgdNesterov(int iters = 1000, double learning_rate = 0.01, int mini_batch = 8, double partion_save_grade = 0.01);

	//Сингулярное разложение матриц (итеративное)
	void svd(Matrix& U, Matrix& s, Matrix& VT);

	~Optimizer();
};