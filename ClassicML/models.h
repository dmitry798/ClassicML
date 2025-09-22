#pragma once
#include "matrix_operations.h"

class LinearRegression
{
private:
	Matrix X;
	Matrix Y;
	Matrix W;

	Matrix X_train;
	Matrix Y_train;
	Matrix X_test;
	Matrix Y_test;

	Matrix X_train_norm;
	Matrix Y_train_norm;
	Matrix X_test_norm;
	Matrix Y_test_norm;

	Matrix mean_x;
	Matrix std_x;
	Matrix mean_y;
	Matrix std_y;

public:

	//конструктор линейной регрессии
	LinearRegression(Matrix x, Matrix y);

	//градиентный спуск
	//learning_rate - скорость обучения
	//epoch - количество эпох
	void train(double learning_rate, int epoch);

	//СВД-разложение...
	Matrix SVD();

	//"предсказания" модели
	Matrix predict();

	//ошибка модели
	double loss();

	//нормализация данных на обучающей и валидационной выборке
	void normalizer();

	//разделение данных из общей выборки на обучающую и валидационную
	void split(double ratio);

	~LinearRegression();
private:

	//высчитывает среднее значение и СКО, вызывает функцию norma
	void transform(const Matrix& Z_train, const Matrix& Z_test, Matrix& Z_train_norm, Matrix& Z_test_norm, Matrix& mean_z, Matrix& std_z);

	//нормализация данных
	Matrix norma(const Matrix& z, const Matrix& mean_z, const Matrix& std_z);

	//денормализация данных для "предсказаний"
	Matrix denorma(const Matrix& z, const Matrix& mean_z, const Matrix& std_z);
};


