#pragma once
#include <cmath>
#include "matrxop.h"

class Matrix
{
private:

	int rows, cols, dim;

	string name;

	double* matrix;

public:

	//конструктор по умолчанию
	Matrix();

	//констурктор 1
	//принимает на вход кол-во строк и столбцов
	Matrix(const int rows, const int cols, string name = "");

	//конструктор 2
	//принимает на вход матрицу, кол-во строк и столбцов
	Matrix(double** matrix, int rows, int cols, string name = "");

	//конструктор 3
	//принимает на вход вектор, кол-во строк и столбцов
	Matrix(double* matrix, int rows, int cols, string name = "");

	//конструктор копирования
	//принимает на вход матрицу того же класса
	Matrix(const Matrix& matrix);

	//конструктор выражений
	template<MatrxOp T>
	Matrix(const T& init) : matrix(nullptr), rows(init.getRows()), cols(init.getCols()), dim(init.getDim())
	{
		allocateMemory();
		for (int i = 0; i < dim; i++) this->matrix[i] = init[i];
	}

	//конструктор перемещения
	//принимает на вход r-value матрицу того же класса
	Matrix(Matrix&& other) noexcept;

	//получение кол-ва строк
	int getRows() const;

	//получение кол-ва колонок
	int getCols() const;

	//получение размерности
	int getDim() const;

	//оператор ()
	//принимает на вход i строку и j столбец
	double& operator() (const int i, const int j) const;

	//оператор []
	//принимает на вход i элемент
	double& operator[] (const int i) const;

	//оператор присваивания
	Matrix& operator= (const Matrix& other) noexcept;

	//оператор перемещения
	Matrix& operator= (Matrix&& other) noexcept;

	//транспонирование
	Matrix transpose() const;

	//генерация рандомных значений в матрице в пределах от 0 до 1
	void random();

	//перемешивание строк в матрице
	void randomShuffle(Matrix& other);

	//вывод размерности матрицы
	void reshape() const;

	//вывод матрицы
	void print(string text = "") const;

	//мат ожидание
	Matrix& mean(const Matrix& x);

	//стандартное отклонение
	Matrix& std(const Matrix& x, const Matrix& mean);

	//очистка матрицы
	void clear();

	//длина вектора
	double len();

	//выделение подмножества строк матрицы
	Matrix sliceRow(int start, int end);

	//выделение подмножества столбцов матрицы
	Matrix sliceCols(int start, int end);

	//натуральный логарифм ко всем элементам матрицы
	Matrix logMatrx();

	//сумма всех элементов матрицы
	double sum();

	//все уникальные элементы матрциы
	Matrix unique();

	//деструктор
	~Matrix();

private:
	
	//выделение памяти
	void allocateMemory();

	//освобождение памяти
	void freeMemory();

	//копирование матрицы в класс
	void copyData(double** matrix);

	//копирование вектора в класс
	void copyVector(double* matrix);

	//копирование матриц одного класса
	void copyFrom(const Matrix& other);

	//заполнение матрицы нулями
	void zeros();
};
