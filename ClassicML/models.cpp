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
	// bias + площадь + количество комнат + этаж
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
		{1.0, 65.0, 2.0, 2.0},   // средняя квартира
		{1.0, 140.0, 4.0, 4.0},  // большая квартира на высоком этаже
		{1.0, 45.0, 1.0, 3.0},   // маленькая квартира
		{1.0, 170.0, 5.0, 1.0},  // большой дом на первом этаже
		{1.0, 105.0, 3.0, 6.0}   // квартира на последнем этаже
	};
	int train_rows_x = 15;
	int train_cols_x = 4;

	srand(time(NULL));

	// Цены в тыс. $
	double yy[][1] = {
		120.5, 180.2, 250.8, 290.3, 350.7,
		200.1, 230.6, 270.9, 310.4, 380.2,
		155.8,  // ожидаемая: ~150-160
		330.4,  // ожидаемая: ~320-340  
		105.2,  // ожидаемая: ~100-110
		365.9,  // ожидаемая: ~360-370
		260.7   // ожидаемая: ~250-270
	};
	int train_rows_y = 15;
	int train_cols_y = 1;

	double** x = copy_static_memory(xx);
	double** y = copy_static_memory(yy);

	Matrix X(x, train_rows_x, train_cols_x, "X");
	Matrix Y(y, train_rows_y, train_cols_y, "Y");

	// Создаем и обучаем модель
	LinearRegression model(X, Y);
	model.split(0.7);
	model.normalizer();
	model.train(0.008, 550);
	model.predict().print();
	cout << model.loss() << endl << endl;
	model.SVD().print();
	cout << model.loss() << endl << endl;


	free_memory_(x, train_rows_x);
	free_memory_(y, train_rows_y);
}




Matrix LinearRegression::SVD()
{
	Matrix X_centred(X_train.get_rows(), X_train.get_cols(), "X_centred");
	Matrix X_test_centred(X_test.get_rows(), X_test.get_cols(), "X_centred");
	Matrix Y_centred(Y_train.get_rows(), Y_train.get_cols(), "Y_centred");
	mean_x.mean(X_train);
	Matrix sum_P(X_train.get_rows(), X_train.get_cols(), "sum_P");

	//центровка X_train
	for (int i = 0; i < X_train.get_cols(); i++)
	{
		for (int j = 0; j < X_train.get_rows(); j++)
		{
			X_centred(j, i) = X_train(j, i) - mean_x(i, 0);
		}
	}
	//центровка X_test_train
	for (int i = 0; i < X_test.get_cols(); i++)
	{
		for (int j = 0; j < X_test.get_rows(); j++)
		{
			X_test_centred(j, i) = X_test(j, i) - mean_x(i, 0);
		}
	}
	//центровка Y_train
	for (int i = 0; i < Y_train.get_cols(); i++)
	{
		for (int j = 0; j < Y_train.get_rows(); j++)
		{
			Y_centred(j, i) = Y_train(j, i) - mean_y(i, 0);
		}
	}
	Matrix X_current = X_centred;
	int comp = std::min(X_train.get_rows(), X_train.get_cols());
	int max_comp = std::min(X_train.get_rows(), X_train.get_cols());
	U = Matrix(X_train.get_rows(), max_comp, "U");
	s = Matrix(max_comp, max_comp, "S");
	VT = Matrix(max_comp, X_train.get_cols(), "VT");
	int k = 0;

	//"обучение"
	while (X_current.len() > 1e-10 && comp > 0)
	{
		Matrix a(1, X_train.get_cols(), "a");
		a.random();
		Matrix b(X_train.get_rows(), 1, "b");
		a = a / a.len();

		double F = 1e10;
		double F_prev = 2e10;

		while ((F_prev - F) / F > 0.000001)
		{
			F_prev = F;
			for (int i = 0; i < X_current.get_rows(); i++)
			{
				double a_quad = 0.0;
				b(i, 0) = 0;
				for (int j = 0; j < X_current.get_cols(); j++)
				{
					a_quad += a(0, j) * a(0, j);
					b(i, 0) += X_current(i, j) * a(0, j);
				}
				b(i, 0) /= a_quad;
			}

			for (int i = 0; i < X_current.get_cols(); i++)
			{
				double b_quad = 0.0;
				a(0, i) = 0.0;
				for (int j = 0; j < X_current.get_rows(); j++)
				{
					b_quad += b(j, 0) * b(j, 0);
					a(0, i) += X_current(j, i) * b(j, 0);
				}
				a(0, i) /= b_quad;
			}
			Matrix P = b * a;
			Matrix Error = X_current - P;
			double er = Error.len();
			F = 0.5 * er * er;
		}
		Matrix u_k = b / b.len();
		Matrix v_k = a / a.len();

		// Заполняем U, S, VT
		for (int i = 0; i < U.get_rows(); i++) {
			U(i, k) = u_k(i, 0);
		}

		s(k, k) = 1 / (b.len() * a.len() + 10e-2) ;
		for (int j = 0; j < VT.get_cols(); j++) {
			VT(k, j) = v_k(0, j);
		}
		Matrix P = b * a;
		sum_P = sum_P + P;
		X_current = X_current - P;
		comp--;
		k++;
	}
	W = VT.transpose() * s * U.transpose() * Y_centred;
	Matrix result = X_test_centred * W;
	for (int i = 0; i < result.get_cols(); i++) {
		for (int j = 0; j < result.get_rows(); j++) {
			result(j, i) += mean_y(i, 0);
		}
	}
	return result;
}