#include "../include/ClassicML/optimization.h"
#include "../include/ClassicML/macros.h"

/************************************************************************************************************************************/

Matrix sigmoid(Matrix&& XW)
{
	for (int i = 0; i < XW.getDim(); i++)
	{
		XW[i] = 1 / (1 + exp(-XW[i]));
	}
	return XW;
}

Matrix softMax(Matrix&& XW)
{
	for (int i = 0; i < XW.getRows(); i++)
	{
		double sum = 0.0;

		for (int j = 0; j < XW.getCols(); j++)
		{
			XW(i, j) = exp(XW(i, j));
			sum += XW(i, j);
		}
		for (int k = 0; k < XW.getCols(); ++k)
			XW(i, k) /= sum;
	}
	return XW;
}

/************************************************************************************************************************************/

/*

Расшифровки номеров методов:

	- LinearRegression: 1
	- LogisticRegression: 2
	- MultiClassLogisticRegression: 3

*/

Optimizer::Optimizer(Dataset& shareData) : data(shareData) {}

void Optimizer::gradientDescent(int iters, double learning_rate, int method)
{
	double alpha = learning_rate;

	Matrix gradient;

	while (iters > 0)
	{
		switch (method)
		{
			case 0:
				std::exception("Choice any method");
				break;

			//- LinearRegression: 1
			case 1:
				gradient = X_train_norm.transpose() * (X_train_norm * W - Y_train_norm) * 2.0;
				break;

			//- LogisticRegression: 2
			case 2:
				gradient = X_train_norm.transpose() * (sigmoid(X_train_norm * W) - Y_train);
				break;

			//- MultiClassLogisticRegression : 3
			case 3:
				gradient = X_train_norm.transpose() * (softMax(X_train_norm * W) - Y_train);
				break;
		}
		W = W - gradient * alpha;
		iters--;
	}
}

void Optimizer::sgd(int iters, double learning_rate, int mini_batch, int method)
{
	double alpha = learning_rate;

	Matrix gradient;

	while (iters > 0)
	{
		for (int start = 0; start < X_train_norm.getRows(); start += mini_batch)
		{
			int end = std::min(start + mini_batch, X_train_norm.getRows());

			Matrix X_batch = X_train_norm.sliceRow(start, end);
			Matrix Y_batch;

			switch (method)
			{
				case 0:
					std::exception("Choice any method");
					break;

				//- LinearRegression: 1
				case 1:
					Y_batch = Y_train_norm.sliceRow(start, end);
					gradient = X_batch.transpose() * (X_batch * W - Y_batch) * 2.0 / (end - start);
					break;

				//- LogisticRegression: 2
				case 2:
					Y_batch = Y_train.sliceRow(start, end);
					gradient = X_batch.transpose() * (sigmoid(X_batch * W) - Y_batch) / (end - start);
					break;

				//- MultiClassLogisticRegression : 3
				case 3:
					Y_batch = Y_train.sliceRow(start, end);
					gradient = X_batch.transpose() * (softMax(X_batch * W) - Y_batch) / (end - start);
					break;
			}
			W = W - gradient * alpha;
		}
		iters--;
	}
}

void Optimizer::sgdMomentum(int iters, double learning_rate, int mini_batch, double partion_save_grade, int method)
{
	Matrix U(W.getRows(), W.getCols(), "U");
	Matrix gradient;

	double gamma = partion_save_grade;
	double alpha = learning_rate;

	while (iters > 0)
	{
		for (int start = 0; start < X_train_norm.getRows(); start += mini_batch)
		{
			int end = std::min(start + mini_batch, X_train_norm.getRows());

			Matrix X_batch = X_train_norm.sliceRow(start, end);
			Matrix Y_batch = Y_train.sliceRow(start, end);

			switch (method)
			{
			case 0:
				std::exception("Choice any method");
				break;

				//- LinearRegression: 1
			case 1:
				Y_batch = Y_train_norm.sliceRow(start, end);
				gradient = X_batch.transpose() * (X_batch * W - Y_batch) * 2.0 / (end - start);
				break;

				//- LogisticRegression: 2
			case 2:
				Y_batch = Y_train.sliceRow(start, end);
				gradient = X_batch.transpose() * (sigmoid(X_batch * W) - Y_batch) / (end - start);
				break;

				//- MultiClassLogisticRegression : 3
			case 3:
				Y_batch = Y_train.sliceRow(start, end);
				gradient = X_batch.transpose() * (softMax(X_batch * W) - Y_batch) / (end - start);
				break;
			}
			U = U * gamma + gradient * alpha;
			W = W - U;
		}
		iters--;
	}
}

void Optimizer::sgdNesterov(int iters, double learning_rate, int mini_batch, double partion_save_grade, int method)
{
	Matrix U(W.getRows(), W.getCols(), "U");
	Matrix gradient;

	double gamma = partion_save_grade;
	double alpha = learning_rate;

	while (iters > 0)
	{
		//вот тут помог исправленный концепт... отчасти
		for (int start = 0; start < X_train_norm.getRows(); start += mini_batch)
		{
			int end = std::min(start + mini_batch, X_train_norm.getRows());

			Matrix X_batch = X_train_norm.sliceRow(start, end);
			Matrix Y_batch = Y_train.sliceRow(start, end);

			switch (method)
			{
				case 0:
					std::exception("Choice any method");
					break;

				//- LinearRegression: 1
				case 1:
					Y_batch = Y_train_norm.sliceRow(start, end);
					gradient = X_batch.transpose() * (X_batch * (W - U * gamma) - Y_batch) * 2.0 / (end - start);
					break;

				//- LogisticRegression: 2
				case 2:
					Y_batch = Y_train.sliceRow(start, end);
					gradient = X_batch.transpose() * (sigmoid(X_batch * (W - U * gamma)) - Y_batch) / (end - start);
					break;

				//- MultiClassLogisticRegression : 3
				case 3:
					Y_batch = Y_train.sliceRow(start, end);
					gradient = X_batch.transpose() * (softMax(X_batch * (W - U * gamma)) - Y_batch) / (end - start);
					break;
			}
			U = U * gamma + gradient * alpha;
			W = W - U;
		}
		iters--;
	}
}

void Optimizer::svd(Matrix& U, Matrix& s, Matrix& VT)
{
	Matrix X_current(X_train.getRows(), X_train.getCols(), "X_centred");
	mean_x.mean(X_train);
	//центровка X_train
	for (int i = 0; i < X_train.getCols(); i++)
	{
		for (int j = 0; j < X_train.getRows(); j++)
		{
			X_current(j, i) = X_train(j, i) - mean_x[i];
		}
	}

	int comp = std::min(X_train.getRows(), X_train.getCols());

	//разложение матрицы X_train на 3 матрицы
	U = Matrix(X_train.getRows(), comp, "U");
	s = Matrix(comp, comp, "S");
	VT = Matrix(comp, X_train.getCols(), "VT");

	Matrix a(1, X_train.getCols(), "a");
	Matrix b(X_train.getRows(), 1, "b");

	int iter = 0;

	//"обучение"
	while (X_current.lenVec() > 1e-10 && comp > 0)
	{
		a.clear(); b.clear();
		a.random();
		a = a / a.lenVec();

		double F = 1e10;
		double F_prev = 2e10;

		while ((F_prev - F) / F > 1e-10)
		{
			F_prev = F;
			for (int i = 0; i < X_current.getRows(); i++)
			{
				double a_quad = 0.0;
				b[i] = 0;
				for (int j = 0; j < X_current.getCols(); j++)
				{
					a_quad += a[j] * a[j];
					b[i] += X_current(i, j) * a[j];
				}
				b[i] /= a_quad;
			}

			for (int i = 0; i < X_current.getCols(); i++)
			{
				double b_quad = 0.0;
				a[i] = 0.0;
				for (int j = 0; j < X_current.getRows(); j++)
				{
					b_quad += b[j] * b[j];
					a[i] += X_current(j, i) * b[j];
				}
				a[i] /= b_quad;
			}
			//						   X - P
			double er = Matrix(X_current - b * a).lenVec();
			F = 0.5 * er * er;
		}

		double sigma = b.lenVec() * a.lenVec();

		//заполняем U, S, VT
		for (int i = 0; i < U.getRows(); i++)
			U(i, iter) = b[i] / b.lenVec();

		//ridge-regular
		if (sigma < 1e-5)
			s(iter, iter) = sigma / (sigma * sigma + 10e-5);
		else
			s(iter, iter) = 1.0 / sigma;

		for (int j = 0; j < VT.getCols(); j++)
			VT(iter, j) = a[j] / a.lenVec();

		//		X = X - P
		X_current = X_current - b * a;
		comp--;
		iter++;
	}
}

Optimizer::~Optimizer(){}


Distance::Distance(Dataset& shareData): data(shareData) {}

Matrix Distance::manhattan(Matrix&& feature)
{
	Matrix result(X_train.getRows(), 1);
	for (int i = 0; i < X_train.getRows(); i++)
	{
		double sum = 0.0;
		for (int j = 0; j < X_train.getCols(); j++)
			sum += abs(feature[j] - X_train_norm(i, j));
		result[i] = sum;
	}
	return result;
}

Matrix Distance::evklid(Matrix&& feature)
{
	Matrix result(X_train.getRows(), 1);
	for (int i = 0; i < X_train.getRows(); i++)
	{
		double sum = 0.0;
		for (int j = 0; j < X_train.getCols(); j++)
		{
			double diff = 0.0;
			diff += feature[j] - X_train_norm(i, j);
			sum += diff * diff;
		}
		result[i] = sqrt(sum);
	}
	return result;
}