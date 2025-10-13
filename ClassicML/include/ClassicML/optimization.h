#pragma once
#include "matrix.h"
#include "preprocessor.h"
using namespace Data;

//сигмоида
Matrix sigmoid(Matrix&& XW);

//soft max
Matrix softMax(Matrix&& XW);

class Optimizer
{
private:
	Dataset& data;

public:

	Optimizer(Dataset& shareData);

	//Градиетный спуск
	void gradientDescent(int iters = 1000, double learning_rate = 0.01, int method = 0);

	//Стохастический градиетный спуск (mini batch)
	void sgd(int iters = 1000, double learning_rate = 0.01, int mini_batch = 8, int method = 0);

	//Стохастический градиетный спуск методом Нестерова (mini batch)
	void sgdNesterov(int iters = 1000, double learning_rate = 0.01, int mini_batch = 8, double partion_save_grade = 0.01, int method = 0);

	//Сингулярное разложение матриц (итеративное)
	void svd(Matrix& U, Matrix& s, Matrix& VT);

	~Optimizer();
};