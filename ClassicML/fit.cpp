#include "fit.h"

Fit::Fit(Data& shareData): data(shareData){}

Matrix Fit::svd()
{
	Matrix X_current(data.X_train.getRows(), data.X_train.getCols(), "X_centred");
	Matrix sum_P(data.X_train.getRows(), data.X_train.getCols(), "X_centred");
	data.mean_x.mean(data.X_train);
	//центровка X_train
	for (int i = 0; i < data.X_train.getCols(); i++)
	{
		for (int j = 0; j < data.X_train.getRows(); j++)
		{
			X_current(j, i) = data.X_train(j, i) - data.mean_x(i, 0);
		}
	}

	int comp = std::min(data.X_train.getRows(), data.X_train.getCols());

	//разложение матрицы X_train на 3 матрицы
	Matrix U(data.X_train.getRows(), comp, "U");
	Matrix s(comp, comp, "S");
	Matrix VT(comp, data.X_train.getCols(), "VT");

	Matrix a(1, data.X_train.getCols(), "a");
	Matrix b(data.X_train.getRows(), 1, "b");

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
				b(i, 0) = 0;
				for (int j = 0; j < X_current.getCols(); j++)
				{
					a_quad += a(0, j) * a(0, j);
					b(i, 0) += X_current(i, j) * a(0, j);
				}
				b(i, 0) /= a_quad;
			}

			for (int i = 0; i < X_current.getCols(); i++)
			{
				double b_quad = 0.0;
				a(0, i) = 0.0;
				for (int j = 0; j < X_current.getRows(); j++)
				{
					b_quad += b(j, 0) * b(j, 0);
					a(0, i) += X_current(j, i) * b(j, 0);
				}
				a(0, i) /= b_quad;
			}
			//					 X - P
			double er = (X_current - b * a).len();
			F = 0.5 * er * er;
		}

		double sigma = b.len() * a.len();

		//заполняем U, S, VT
		for (int i = 0; i < U.getRows(); i++)
			U(i, iter) = b(i, 0) / b.len();

		//ridge-regular
		if (sigma < 1e-5)
			s(iter, iter) = sigma / (sigma * sigma + 10e-5);
		else
			s(iter, iter) = 1 / sigma;

		for (int j = 0; j < VT.getCols(); j++)
			VT(iter, j) = a(0, j) / a.len();

		//		X = X - P
		X_current = X_current - b * a;
		comp--;
		iter++;
		sum_P = sum_P + b * a;
	}
	W = VT.transpose() * s * U.transpose() * data.Y_train;
	return W;
}