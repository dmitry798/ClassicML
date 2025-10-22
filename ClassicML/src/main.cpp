#include "../include/ClassicML.h"

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
	int train_rows_x = 18;
	int train_cols_x = 4;  // [часы_учёбы, часы_сна, посещаемость, сложность_предмета]
	double xx[][4] = {
		// class 0 (label = 0)
		{0.8f, 0.9f, 1.0f, 0.0f},
		{1.1f, 1.0f, 1.0f, 0.0f},
		{1.2f, 0.7f, 1.0f, 0.0f},
		{0.9f, 1.3f, 1.0f, 0.0f},
		{1.3f, 1.1f, 1.0f, 0.0f},
		{0.7f, 0.8f, 1.0f, 0.0f},

		// class 1 (label = 1)
		{4.7f, 0.9f, 1.0f, 1.0f},
		{5.2f, 1.2f, 1.0f, 1.0f},
		{5.1f, 0.7f, 1.0f, 1.0f},
		{4.8f, 1.3f, 1.0f, 1.0f},
		{5.3f, 1.0f, 1.0f, 1.0f},
		{4.9f, 0.8f, 1.0f, 1.0f},

		// class 2 (label = 2)
		{2.7f, 3.7f, 1.0f, 2.0f},
		{3.3f, 4.2f, 1.0f, 2.0f},
		{3.1f, 3.8f, 1.0f, 2.0f},
		{2.9f, 4.1f, 1.0f, 2.0f},
		{3.4f, 3.9f, 1.0f, 2.0f},
		{2.8f, 4.3f, 1.0f, 2.0f}
	};

	int train_rows_y = 18;
	int train_cols_y = 1;
	double yy[][1] = {
		{1}, {1}, {1}, {1}, {1}, {1},
		{2}, {2}, {2}, {2}, {2}, {2},
		{3}, {3}, {3}, {3}, {3}, {3}
	};

	double** x = copy_static_memory(xx);
	double** y = copy_static_memory(yy);

	Matrix X(x, train_rows_x, train_cols_x, "X");
	Matrix Y(y, train_rows_y, train_cols_y, "Y");

	Y = OneHotEncoder(Y);

	// Предобработка
	Dataset data(X, Y);
	StandartScaler scaler(data);
	scaler.split();
    scaler.standartNormalize();

	// Создаем и обучаем модель
	Knn model(data, 4);
	model.predict().print();
	/*model.loss(0.99);*/
	data.info();


	free_memory_(x, train_rows_x);
	free_memory_(y, train_rows_y);

	return 0;
}