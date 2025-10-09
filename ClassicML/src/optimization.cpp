#include "../include/ClassicML/optimization.h"
#include "../include/ClassicML/macros.h"

Optimizer::Optimizer(Dataset& shareData): data(shareData){}

void Optimizer::stochastic_gradient_descent(double learning_rate, int epoch)
{

}

void Optimizer::nesterov(int iters, double learning_rate, double partion_save_grade)
{
	Matrix U(W.getRows(), W.getCols(), "U");
	Matrix gradient;
	Matrix XT = X_train_norm.transpose();

	double gamma = partion_save_grade;
	double alpha = learning_rate;

	while (iters > 0)
	{
		//вот тут помог исправленный концепт... отчасти
		gradient = XT * (X_train_norm * (W - U * gamma) - Y_train_norm) * 2;
		U = U * gamma + gradient * alpha;
		W = W - U;
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
	while (X_current.len() > 1e-10 && comp > 0)
	{
		a.clear(); b.clear();
		a.random();
		a = a / a.len();

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
			double er = Matrix(X_current - b * a).len();
			F = 0.5 * er * er;
		}

		double sigma = b.len() * a.len();

		//заполняем U, S, VT
		for (int i = 0; i < U.getRows(); i++)
			U(i, iter) = b[i] / b.len();

		//ridge-regular
		if (sigma < 1e-5)
			s(iter, iter) = sigma / (sigma * sigma + 10e-5);
		else
			s(iter, iter) = 1 / sigma;

		for (int j = 0; j < VT.getCols(); j++)
			VT(iter, j) = a[j] / a.len();

		//		X = X - P
		X_current = X_current - b * a;
		comp--;
		iter++;
	}
}

Optimizer::~Optimizer()
{

}