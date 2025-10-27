#pragma once
#include <cmath>
#include "matrxop.h"
using std::unique_ptr;
using std::swap;
using std::make_unique;
using std::copy;

#include <type_traits>
#include <utility>

// has getRows()
template<typename T>
struct has_getRows {
	template<typename U> static auto test(int) -> decltype(std::declval<const U&>().getRows(), std::true_type{});
	template<typename>  static auto test(...) -> std::false_type;
	static constexpr bool value = decltype(test<T>(0))::value;
};

// has getCols()
template<typename T>
struct has_getCols {
	template<typename U> static auto test(int) -> decltype(std::declval<const U&>().getCols(), std::true_type{});
	template<typename>  static auto test(...) -> std::false_type;
	static constexpr bool value = decltype(test<T>(0))::value;
};

// has getDim()
template<typename T>
struct has_getDim {
	template<typename U> static auto test(int) -> decltype(std::declval<const U&>().getDim(), std::true_type{});
	template<typename>  static auto test(...) -> std::false_type;
	static constexpr bool value = decltype(test<T>(0))::value;
};

// has operator[](int)
template<typename T>
struct has_brackets_index {
	template<typename U> static auto test(int) -> decltype(std::declval<const U&>()[0], std::true_type{});
	template<typename>  static auto test(...) -> std::false_type;
	static constexpr bool value = decltype(test<T>(0))::value;
};

// has operator()(int,int)
template<typename T>
struct has_paren_ij {
	template<typename U> static auto test(int) -> decltype(std::declval<const U&>()(0, 0), std::true_type{});
	template<typename>  static auto test(...) -> std::false_type;
	static constexpr bool value = decltype(test<T>(0))::value;
};

class Matrix
{
private:

	int rows, cols, dim;

	string name;

	unique_ptr<double[]> matrix{nullptr};

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
	Matrix(const Matrix& other);

	//конструктор выражений
	template<typename T,
		typename std::enable_if<
		has_getRows<T>::value&&
		has_getCols<T>::value&&
		has_getDim<T>::value&&
		has_brackets_index<T>::value, int>::type = 0>
	Matrix(const T& init)
		: matrix(nullptr), rows(init.getRows()), cols(init.getCols()), dim(init.getDim())
	{
		allocateMemory();
		for (int i = 0; i < dim; ++i)
			this->matrix[i] = init[i];
	}

	//конструктор перемещения
	//принимает на вход r-value матрицу того же класса
	Matrix(Matrix&&) noexcept;

	//получение кол-ва строк
	int getRows() const;

	//получение кол-ва колонок
	int getCols() const;

	//получение размерности
	int getDim() const;

	double* getData() const;

	//оператор ()
	//принимает на вход i строку и j столбец
	double& operator() (const int i, const int j) const;

	//оператор []
	//принимает на вход i элемент
	double& operator[] (const int i) const;

	//оператор присваивания
	Matrix& operator= (const Matrix& other);

	//оператор перемещения
	Matrix& operator= (Matrix&&) noexcept;

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

	//очистка матрицы
	void clear();

	//длина вектора
	double lenVec();

	//выделение подмножества строк матрицы
	Matrix sliceRow(int start, int end);

	//выделение подмножества столбцов матрицы
	Matrix sliceCols(int start, int end);

	//натуральный логарифм ко всем элементам матрицы
	Matrix logMatrx();

	//сумма всех элементов матрицы
	double sum();

	//все уникальные элементы матрицы
	Matrix unique();

	//сортировка по строкам с указанием по какому столбцу сортировать
	Matrix sortRows(int t) const;

	//копирование матриц одного класса
	void copyFrom(const Matrix& other);

	////округление элементов матрицы
	//Matrix roundMatrx();

	////корень от всех элементов матрицы
	//Matrix sqrtMatrx();

	////модуль от всех элементов матрицы
	//Matrix absMatrx();

	//деструктор
	~Matrix();

	//заполнение матрицы нулями
	void zeros();

private:
	
	//выделение памяти
	void allocateMemory();

	//копирование матрицы в класс
	void copyData(double** matrix);

	//копирование вектора в класс
	void copyVector(double* matrix);
};

//мат ожидание
Matrix mean(const Matrix& x);

//стандартное отклонение
Matrix stddev(const Matrix& x, const Matrix& mean);

//сортировка матриц
Matrix sort(Matrix& matrix);

//мода элементов
int mode(const Matrix& col);