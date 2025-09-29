#include <iostream>
#include "models.h"

Models::Models(Data& shareData) : data(shareData), fit(data), W(data.X.getCols(), data.Y.getCols(), "W")
{
	W.random();
}

LinearRegression::LinearRegression(Data& shareData):Models(shareData) {}

void LinearRegression::train()
{
	Matrix U;
	Matrix s;
	Matrix VT;

	fit.svd(U, s, VT);

	W = VT.transpose() * s * U.transpose() * data.Y_train;
}

Matrix LinearRegression::predict() const
{
	return data.X_test * W;
}

Matrix LinearRegression::predict(Matrix& X_predict) const
{
	return X_predict * W;
}

LinearRegression::~LinearRegression(){}

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
		{1.0, 65.0, 2.0, 2.0},
		{1.0, 140.0, 4.0, 4.0},
		{1.0, 45.0, 1.0, 3.0},
		{1.0, 170.0, 5.0, 1.0},
		{1.0, 105.0, 3.0, 6.0}
	};
	int train_rows_x = 15;
	int train_cols_x = 4;

	double yy[][1] = {
		120.5, 180.2, 250.8, 290.3, 350.7,
		200.1, 230.6, 270.9, 310.4, 380.2,
		155.8,  //~150-160
		330.4,  //~320-340  
		105.2,  //~100-110
		365.9,  //~360-370
		260.7   //~250-270
	};
	int train_rows_y = 15;
	int train_cols_y = 1;

	double** x = copy_static_memory(xx);
	double** y = copy_static_memory(yy);

	Matrix X(x, train_rows_x, train_cols_x, "X");
	Matrix Y(y, train_rows_y, train_cols_y, "Y");

	// Предобработка
	Data data(X, Y);
	StandartScaler scaler(data);
	scaler.split(0.7);
	
	// Создаем и обучаем модель
	LinearRegression model(data);
	model.train();
	data.Y_test.print();
	model.predict().print();

	free_memory_(x, train_rows_x);
	free_memory_(y, train_rows_y);

	return 0;
}