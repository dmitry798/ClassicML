#include <iostream>
#include "models.h"
using std::cin;
using std::cout;

LinearRegression::LinearRegression(Matrix x, Matrix y): 
	X(move(x)), 
	Y(move(y)),
	W(X.get_cols(), Y.get_cols(), "W"),
	mean_x(X.get_cols(), 1, "mean_x"),
	std_x(X.get_cols(), 1, "std_x"),
	mean_y(Y.get_cols(), 1, "mean_y"),
	std_y(Y.get_cols(), 1, "std_y")
{
	W.random();
}

double LinearRegression::loss()
{
	Matrix error = X_train_norm * W - Y_train_norm;
	return (error.transpose() * error)(0, 0) / X_train_norm.get_rows();
}

void LinearRegression::train(double learning_rate, int epoch)
{
	int i = 0;
	Matrix XT = X_train_norm.transpose();
	while (i != epoch)
	{
		Matrix gradient = XT * (X_train_norm * W - Y_train_norm);
		W = W - gradient * (learning_rate * 2.0 / X_train_norm.get_rows());
		i++;
	}
}

Matrix LinearRegression::predict()
{
	Matrix normalized_pred = X_test_norm * W;
	return denorma(normalized_pred, mean_y, std_y);
}

void LinearRegression::normalizer()
{
	transform(X_train, X_test, X_train_norm, X_test_norm, mean_x, std_x);
	transform(Y_train, Y_test, Y_train_norm, Y_test_norm, mean_y, std_y);
	X_test.print(); Y_test.print();
}

void LinearRegression::split(double ratio)
{
	//X.random_shuffle(Y);
	int rows_train = (int)(X.get_rows() * ratio);
	int rows_test = X.get_rows() - rows_train;

	X_train = Matrix(rows_train, X.get_cols());
	Y_train = Matrix(rows_train, Y.get_cols());

	X_test = Matrix(rows_test, X.get_cols());
	Y_test = Matrix(rows_test, Y.get_cols());

	for (int i = 0; i < rows_train; i++)
	{
		for (int j = 0; j < X.get_cols(); j++)
		{
			X_train(i, j) = X(i, j);
		}

		for (int j = 0; j < Y.get_cols(); j++)
		{
			Y_train(i, j) = Y(i, j);
		}
	}

	int iter = 0;
	for (int i = rows_train; i < X.get_rows(); i++)
	{
		for (int j = 0; j < X.get_cols(); j++)
		{
			X_test(iter, j) = X(i, j);
		}

		for (int j = 0; j < Y.get_cols(); j++)
		{
			Y_test(iter, j) = Y(i, j);
		}
		iter++;
	}
}

Matrix LinearRegression::norma(const Matrix& z, const Matrix& mean_z, const Matrix& std_z)
{
	Matrix normalized_z = z;
	int cols = z.get_cols();
	int rows = z.get_rows();

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

Matrix LinearRegression::denorma(const Matrix& z, const Matrix& mean_z, const Matrix& std_z)
{
	Matrix denormalized_z = z;
	int cols = z.get_cols();
	int rows = z.get_rows();

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

void LinearRegression::transform(const Matrix& Z_train, const Matrix& Z_test, Matrix& Z_train_norm, Matrix& Z_test_norm, Matrix& mean_z, Matrix& std_z)
{
	mean_z.mean(Z_train);
	std_z.std(Z_train, mean_z);

	Z_train_norm = Matrix(Z_train.get_rows(), Z_train.get_cols(), "train_norm");
	Z_train_norm = norma(Z_train, mean_z, std_z);

	Z_test_norm = Matrix(Z_test.get_rows(), Z_test.get_cols(), "test_norm");
	Z_test_norm = norma(Z_test, mean_z, std_z);
}


LinearRegression::~LinearRegression()
{

}

//void LinearRegression::SVD()
//{
//	int rows = W.get_rows();
//	int cols = W.get_cols();
//
//	Matrix V(rows, rows, "V");
//	Matrix E(rows, cols, "E");
//	Matrix U(cols, cols, "U");
//
//	W.transpons()* W;
//}

template <int R, int C>
double** copy_static_memory(double(&matrix)[R][C])
{
	double** dest = new double* [R];
	for (int i = 0; i < R; i++)
	{
		dest[i] = new double[C];
	}
	for (int i = 0; i < R; i++)
	{
		for (int j = 0; j < C; j++)
		{
			dest[i][j] = matrix[i][j];
		}
	}
	return dest;
}

static void free_memory_(double** matrix, int rows)
{
	for (int i = 0; i < rows; i++)
	{
		delete[] matrix[i];
	}
	delete[] matrix;
	matrix = nullptr;
}



int main() 
{
	// bias + ������� + ���������� ������ + ����
	double xx[][4] = {
		{1.0, 50.0, 1.0, 1.0},
		{1.0, 75.0, 2.0, 1.0},
		{1.0, 100.0, 3.0, 2.0},
		{1.0, 120.0, 3.0, 3.0},
		{1.0, 150.0, 4.0, 2.0},
		{1.0, 80.0, 2.0, 5.0},
		{1.0, 95.0, 3.0, 1.0},
		{1.0, 110.0, 3.0, 4.0},
		{1.0, 130.0, 4.0, 3.0},
		{1.0, 160.0, 5.0, 2.0},
		{1.0, 65.0, 2.0, 2.0},   // ������� ��������
		{1.0, 140.0, 4.0, 4.0},  // ������� �������� �� ������� �����
		{1.0, 45.0, 1.0, 3.0},   // ��������� ��������
		{1.0, 170.0, 5.0, 1.0},  // ������� ��� �� ������ �����
		{1.0, 105.0, 3.0, 6.0}   // �������� �� ��������� �����
	};
	int train_rows_x = 15;
	int train_cols_x = 4;

	srand(time(NULL));

	// ���� � ���. $
	double yy[][1] = {
		120.5, 180.2, 250.8, 290.3, 350.7,
		200.1, 230.6, 270.9, 310.4, 380.2,
		155.8,  // ���������: ~150-160
		330.4,  // ���������: ~320-340  
		105.2,  // ���������: ~100-110
		365.9,  // ���������: ~360-370
		260.7   // ���������: ~250-270
	};
	int train_rows_y = 15;
	int train_cols_y = 1;

	double** x = copy_static_memory(xx);
	double** y = copy_static_memory(yy);

	Matrix X(x, train_rows_x, train_cols_x, "X");
	Matrix Y(y, train_rows_y, train_cols_y, "Y");

	// ������� � ������� ������
	LinearRegression model(X, Y);
	model.split(0.7);
	model.normalizer();
	/*model.train(0.01, 55);
	model.predict().print();
	cout << model.loss() << endl;*/
	model.SVD().print();


	free_memory_(x, train_rows_x);
	free_memory_(y, train_rows_y);
}




Matrix LinearRegression::SVD()
{
	Matrix a(X_train.get_cols(), 1, "a");
	a.random();
	Matrix b(1, X_train.get_rows(), "b");
	b.random();
	Matrix X_centred(X_train.get_rows(), X_train.get_cols(), "X_centred");
	mean_x.mean(X_train);
	Matrix P(X_train.get_rows(), X_train.get_cols(), "P");
	Matrix sum_P(X_train.get_rows(), X_train.get_cols(), "sum_P");
	Matrix Error(X_train.get_rows(), X_train.get_cols(), "Error");
	double F = 1.0;

	//��������� X_train
	for (int i = 0; i < X_train.get_cols(); i++)
	{
		for (int j = 0; j < X_train.get_rows(); j++)
		{
			double a = X_train(j, i) - mean_x(i, 0);
			X_centred(j, i) = X_train(j, i) - mean_x(i, 0);
		}
	}
	Matrix X_current = X_centred;
	//"��������"
	while (F > 0.01)
	{
		for (int i = 0; i < X_current.get_rows(); i++)
		{
			double a_quad = 0.0;
			for (int j = 0; j < X_current.get_cols(); j++)
			{
				a_quad += a(j, 0) * a(j, 0);
				b(0, i) += X_current(i, j) * a(j, 0);
			}
			b(0, i) /= a_quad;
		}

		for (int i = 0; i < X_current.get_cols(); i++)
		{
			double b_quad = 0.0;
			for (int j = 0; j < X_current.get_rows(); j++)
			{
				b_quad += b(0, j) * b(0, j);
				a(i, 0) += X_current(j, i) * b(0, j);
			}
			a(i, 0) /= b_quad;
		}

		P = a * b;
		X_current = X_current - P;
		Error = X_current - P;
		int er = Error.len();
		F = abs(0.5 * er * er);
		sum_P = sum_P + P;
	}
	return sum_P;
}