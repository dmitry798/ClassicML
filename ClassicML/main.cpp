#include "ClassicML.h"

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
    {1.0, 105.0, 3.0, 6.0},
    {1.0, 55.0, 2.0, 2.0},
    {1.0, 85.0, 2.0, 3.0},
    {1.0, 115.0, 4.0, 2.0},
    {1.0, 135.0, 4.0, 5.0},
    {1.0, 175.0, 5.0, 4.0},
    {1.0, 95.0, 3.0, 2.0},
    {1.0, 125.0, 4.0, 1.0},
    {1.0, 145.0, 5.0, 3.0},
    {1.0, 155.0, 5.0, 5.0}
    };

    int train_rows_x = 24;
    int train_cols_x = 4;

    double yy[][3] = {
        {120.5, 50.2, 300.1},
        {180.2, 80.1, 400.0},
        {250.8, 120.5, 500.3},
        {290.3, 140.0, 550.4},
        {350.7, 180.2, 650.7},
        {200.1, 90.1, 350.0},
        {230.6, 110.2, 420.1},
        {270.9, 130.0, 490.2},
        {310.4, 160.5, 580.5},
        {380.2, 210.0, 700.0},
        {155.8, 70.1, 300.0},
        {330.4, 190.2, 600.3},
        {105.2, 40.0, 250.1},
        {365.9, 200.1, 650.4},
        {260.7, 130.0, 480.2},
        {130.0, 60.2, 280.1},
        {190.0, 100.0, 390.0},
        {280.0, 150.0, 520.2},
        {320.0, 170.1, 590.4},
        {390.0, 220.0, 720.0},
        {210.0, 110.0, 400.2},
        {300.0, 160.0, 560.1},
        {340.0, 190.0, 620.3},
        {370.0, 210.0, 670.5}
    };

    int train_rows_y = 24;
    int train_cols_y = 3;

	double** x = copy_static_memory(xx);
	double** y = copy_static_memory(yy);

	Matrix X(x, train_rows_x, train_cols_x, "X");
	Matrix Y(y, train_rows_y, train_cols_y, "Y");

	// Предобработка
	Dataset data(X, Y);
	StandartScaler scaler(data);
	scaler.split(0.7, false);

	// Создаем и обучаем модель
	LinearRegression model(data);
	model.train();
	data.Y_test.print();
	model.predict().print();
	model.loss();


	free_memory_(x, train_rows_x);
	free_memory_(y, train_rows_y);

	return 0;
}