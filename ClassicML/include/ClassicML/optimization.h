#pragma once
#include "matrix.h"
#include "preprocessor.h"
using namespace Data;

//сигмоида
Matrix sigmoid(Matrix&& XW);

class Optimizer
{
private:
	Dataset& data;

public:

	Optimizer(Dataset& shareData);

	//Стохастический градиетный спуск (mini batch)
	void sgd(int iters = 1000, double learning_rate = 0.01, int mini_batch = 8, bool logit = false);

	//Стохастический градиетный спуск методом Нестерова (mini batch)
	void sgdNesterov(int iters = 1000, double learning_rate = 0.01, int mini_batch = 8, double partion_save_grade = 0.01, bool logistic = false);

	//Сингулярное разложение матриц (итеративное)
	void svd(Matrix& U, Matrix& s, Matrix& VT);

	~Optimizer();
};