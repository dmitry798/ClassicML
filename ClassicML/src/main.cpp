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
		// Класс 0: мало учится, мало спит, низкая посещаемость, высокая сложность
		{4.0, 4.0, 62.0, 8.5},
		{5.0, 3.0, 58.0, 9.0},
		{4.5, 4.5, 54.0, 8.0},
		{5.5, 3.5, 59.0, 8.8},
		{3.0, 4.0, 61.0, 9.0},
		{4.5, 3.5, 56.0, 9.2},
		// Класс 1: средне учится, средне спит, хорошая посещаемость, средняя сложность
		{10.0, 7.0, 82.0, 5.5},
		{11.5, 7.5, 85.0, 4.5},
		{10.5, 8.0, 84.0, 5.0},
		{12.0, 8.0, 81.0, 6.0},
		{9.5, 7.0, 80.0, 5.5},
		{11.0, 8.0, 83.0, 5.0},
		// Класс 2: отлично учится, отлично спит, высокая посещаемость, низкая сложность
		{15.0, 9.0, 96.0, 2.0},
		{16.0, 9.5, 98.0, 1.0},
		{15.5, 9.0, 95.0, 2.0},
		{16.5, 8.5, 99.0, 1.0},
		{15.0, 9.5, 94.0, 2.0},
		{14.5, 8.5, 97.0, 1.0}
	};

	int train_rows_y = 18;
	int train_cols_y = 1;
	double yy[][1] = {
		{0}, {0}, {0}, {0}, {0}, {0},
		{1}, {1}, {1}, {1}, {1}, {1},
		{3}, {3}, {3}, {3}, {3}, {3}
	};

	double** x = copy_static_memory(xx);
	double** y = copy_static_memory(yy);

	Matrix X(x, train_rows_x, train_cols_x, "X");
	Matrix Y(y, train_rows_y, train_cols_y, "Y");

	// Предобработка
	Dataset data(X, Y);
	StandartScaler scaler(data);
	scaler.split();
    scaler.standartNormalize();

	// Создаем и обучаем модель
	MultiClassLogisticRegression model(data);
	model.train("sgd", 100, 0.1, 1);
	data.Y_test.print();
	model.predict().print();
	model.loss();


	free_memory_(x, train_rows_x);
	free_memory_(y, train_rows_y);

	return 0;
}