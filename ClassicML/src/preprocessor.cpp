#include "../include/ClassicML/preprocessor.h"

Dataset::Dataset(Matrix& x, Matrix& y) :
	X(move(x)),
	Y(move(y)),
	mean_x(X.getCols(), 1, "mean_x"),
	std_x(X.getCols(), 1, "std_x"),
	mean_y(Y.getCols(), 1, "mean_y"),
	std_y(Y.getCols(), 1, "std_y"),
	W(X.getCols(), Y.getCols(), "W")
{
	srand(time(NULL));
	W.random();
}

void Dataset::info()
{
	Y_test.print("Y_test: ");
	Y_pred.print("Y_predict: ");
}

/************************************************************************************************************************************/


#include "../include/ClassicML/macros.h"

StandartScaler::StandartScaler(Dataset& sharedData) : data(sharedData) {}

void StandartScaler::split(double ratio, bool random)
{
	if (random)
		X.randomShuffle(Y);
	int rows_train = (int)(X.getRows() * ratio);
	int rows_test = X.getRows() - rows_train;

	X_train = Matrix(rows_train, X.getCols(), "X_train");
	Y_train = Matrix(rows_train, Y.getCols(), "Y_train");

	X_test = Matrix(rows_test, X.getCols(), "X_test");
	Y_test = Matrix(rows_test, Y.getCols(), "Y_test");

	for (int i = 0; i < rows_train; i++)
	{
		for (int j = 0; j < X.getCols(); j++)
		{
			X_train(i, j) = X(i, j);
		}

		for (int j = 0; j < Y.getCols(); j++)
		{
			Y_train(i, j) = Y(i, j);
		}
	}

	int iter = 0;
	for (int i = rows_train; i < X.getRows(); i++)
	{
		for (int j = 0; j < X.getCols(); j++)
		{
			X_test(iter, j) = X(i, j);
		}

		for (int j = 0; j < Y.getCols(); j++)
		{
			Y_test(iter, j) = Y(i, j);
		}
		iter++;
	}
}

void StandartScaler::standartNormalize()
{
	transform(X_train, X_test, X_train_norm, X_test_norm, mean_x, std_x);
	transform(Y_train, Y_test, Y_train_norm, Y_test_norm, mean_y, std_y);
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



Matrix OneHotEncoder(Matrix& Z)
{
	int total_cols = 0;

	for (int i = 0; i < Z.getCols(); i++)
	{
		Matrix Z_slice = Z.sliceCols(i, i + 1);
		total_cols += Z_slice.unique().getRows();
	}

	Matrix result(Z.getRows(), total_cols);

	int current_col = 0;
	for (int i = 0; i < Z.getCols(); i++)
	{
		Matrix Z_slice = Z.sliceCols(i, i + 1);
		Matrix unique_vals = Z_slice.unique();
		for (int r = 0; r < Z.getRows(); r++)
		{
			for (int j = 0; j < unique_vals.getRows(); j++)
			{
				if (Z_slice[r] == unique_vals[j])
				{
					result(r, current_col + j) = 1;
					break;
				}
			}
		}
		current_col += unique_vals.getRows();
	}

	return result;
}