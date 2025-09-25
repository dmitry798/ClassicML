#include "preprocessor.h"

Data::Data(Matrix& x, Matrix& y) :
	X(move(x)),
	Y(move(y)),
	mean_x(X.getCols(), 1, "mean_x"),
	std_x(X.getCols(), 1, "std_x"),
	mean_y(Y.getCols(), 1, "mean_y"),
	std_y(Y.getCols(), 1, "std_y")
{
	srand(time(NULL));
}


StandartScaler::StandartScaler(Data& sharedData) : data(sharedData) {}

void StandartScaler::split(double ratio)
{
	data.X.random_shuffle(data.Y);
	int rows_train = (int)(data.X.getRows() * ratio);
	int rows_test = data.X.getRows() - rows_train;

	data.X_train = Matrix(rows_train, data.X.getCols(), "X_train");
	data.Y_train = Matrix(rows_train, data.Y.getCols(), "Y_train");

	data.X_test = Matrix(rows_test, data.X.getCols(), "X_test");
	data.Y_test = Matrix(rows_test, data.Y.getCols(), "Y_test");

	for (int i = 0; i < rows_train; i++)
	{
		for (int j = 0; j < data.X.getCols(); j++)
		{
			data.X_train(i, j) = data.X(i, j);
		}

		for (int j = 0; j < data.Y.getCols(); j++)
		{
			data.Y_train(i, j) = data.Y(i, j);
		}
	}

	int iter = 0;
	for (int i = rows_train; i < data.X.getRows(); i++)
	{
		for (int j = 0; j < data.X.getCols(); j++)
		{
			data.X_test(iter, j) = data.X(i, j);
		}

		for (int j = 0; j < data.Y.getCols(); j++)
		{
			data.Y_test(iter, j) = data.Y(i, j);
		}
		iter++;
	}
}

void StandartScaler::standartNormalize()
{
	transform(data.X_train, data.X_test, data.X_train_norm, data.X_test_norm, data.mean_x, data.std_x);
	transform(data.Y_train, data.Y_test, data.Y_train_norm, data.Y_test_norm, data.mean_y, data.std_y);
}

void StandartScaler::transform(const Matrix& Z_train, const Matrix& Z_test, Matrix& Z_train_norm, Matrix& Z_test_norm, Matrix& mean_z, Matrix& std_z)
{
	mean_z.mean(Z_train);
	std_z.std(Z_train, mean_z);

	Z_train_norm = Matrix(Z_train.getRows(), Z_train.getCols(), "train_norm");
	Z_train_norm = normalize(Z_train, mean_z, std_z);

	Z_test_norm = Matrix(Z_test.getRows(), Z_test.getCols(), "test_norm");
	Z_test_norm = normalize(Z_test, mean_z, std_z);
}

Matrix StandartScaler::normalize(const Matrix& z, const Matrix& mean_z, const Matrix& std_z)
{
	Matrix normalized_z = z;
	int cols = z.getCols();
	int rows = z.getRows();

	double eps = 1e-6;
	for (int i = 0; i < cols; i++)
	{
		for (int j = 0; j < rows; j++)
		{
			normalized_z(j, i) = (z(j, i) - mean_z(i, 0)) / (std_z(i, 0) + eps);
		}
	}
	return normalized_z;
}

Matrix StandartScaler::denormalize(const Matrix& z, const Matrix& mean_z, const Matrix& std_z)
{
	Matrix denormalized_z = z;
	int cols = z.getCols();
	int rows = z.getRows();

	double eps = 1e-6;
	for (int i = 0; i < cols; i++)
	{
		for (int j = 0; j < rows; j++)
		{
			denormalized_z(j, i) = z(j, i) * (std_z(i, 0) + eps) + mean_z(i, 0);
		}
	}
	return denormalized_z;
}
