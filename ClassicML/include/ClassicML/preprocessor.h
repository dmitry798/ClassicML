#pragma once
#include "matrix.h"

namespace Data
{
	struct Dataset
	{
	public:
		Matrix X;
		Matrix Y;
		Matrix W;

		Matrix X_train;
		Matrix Y_train;
		Matrix X_test;
		Matrix Y_test;

		Matrix Y_pred;

		Matrix X_train_norm;
		Matrix Y_train_norm;
		Matrix X_test_norm;
		Matrix Y_test_norm;

		Matrix mean_x;
		Matrix std_x;
		Matrix mean_y;
		Matrix std_y;

		Dataset(Matrix& x, Matrix& y);
	};
}

using namespace Data;

class StandartScaler
{
private:
	
	Dataset& data;

	//высчитывает среднее значение и СКО, вызывает функцию norma
	void transform(const Matrix& Z_train, const Matrix& Z_test, Matrix& Z_train_norm, Matrix& Z_test_norm, Matrix& mean_z, Matrix& std_z);

	//нормализация данных
	Matrix normalize(const Matrix& z, const Matrix& mean_z, const Matrix& std_z);

public:

	StandartScaler(Dataset& sharedData);

	//денормализация данных для "предсказаний"
	static Matrix denormalize(const Matrix& z, const Matrix& mean_z, const Matrix& std_z);

	//нормализация данных на обучающей и валидационной выборке
	void standartNormalize();

	//разделение данных из общей выборки на обучающую и валидационную
	void split(double ratio, bool random);
};