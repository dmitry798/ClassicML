#pragma once
#include <iostream>
#include <cmath>
using std::cin; 
using std::cout; 
using std::endl;
using std::move;
using std::string;

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
	Matrix(const int rows, const int cols, string name = "");

	//конструктор 2
	Matrix(double** matrix, int rows, int cols, string name = "");

	//конструктор копирования
	Matrix(const Matrix& matrix);

	//конструктор перемещения
	Matrix(Matrix&& other) noexcept;

	//получение кол-ва строк
	int get_rows() const;

	//получение кол-ва колонок
	int get_cols() const;

	//получение размерности
	int get_dim() const;

	//оператор ()
	double& operator() (const int i, const int j) const;

	//оператор суммы матриц
	Matrix operator+ (const Matrix& other) const;

	//оператор разности матриц
	Matrix operator- (const Matrix& other) const;

	//оператор умножения матриц
	Matrix operator* (const Matrix& other) const;

	//оператор суммы со скаляром
	Matrix operator+ (double value) const;

	//оператор вычитания скаляра
	Matrix operator- (double value) const;

	//оператор умножения на скаляр
	Matrix operator* (double value) const;

	//оператор деления на скаляр
	Matrix operator/ (double value) const;

	//оператор присваивания
	Matrix& operator= (const Matrix& other) noexcept;

	//оператор перемещения
	Matrix& operator= (Matrix&& other) noexcept;

	//транспонирование
	Matrix transpose() const;

	//генерация рандомных значений в матрице
	void random();

	//перемешивание строк в матрице
	void random_shuffle(Matrix& other);

	//вывод размерности матрицы
	void reshape() const;

	//вывод матрицы
	void print() const;

	//мат ожидание
	Matrix& mean(const Matrix& x);

	//стандартное отклонение
	Matrix& std(const Matrix& x, const Matrix& mean);

	//длина вектора
	double len();

	//деструктор
	~Matrix();

private:
	
	//выделение памяти
	void allocate_memory();

	//освобождение памяти
	void free_memory();

	//копирование матрицы в класс
	void copy_data(double** matrix);

	//копирование матриц одного класса
	void copy_from(const Matrix& other);

	//заполнение матрицы нулями
	void zeros();
};
