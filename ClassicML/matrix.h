#pragma once
#include <iostream>
#include <cmath>
using std::cout; 
using std::endl;
using std::move;
using std::string;

template<typename A>
concept MatrxOp = requires(const A & a, int i, int j) 
{
	{ a[i] } -> std::same_as<double>;
	{ a(i, j) } -> std::same_as<double>;
	{ a.getRows() } -> std::same_as<int>;
	{ a.getCols() } -> std::same_as<int>;
	{ a.getDim() } -> std::same_as<int>;
};

template<typename A, typename B>
struct MatrxSumOp 
{
	MatrxSumOp(const A& a, const B& b) : a_(a), b_(b), rows(a_.getRows()), cols(a_.getCols()) {}

	const A& a_;
	const B& b_;

	int rows;
	int cols;

	int getRows() const { return rows; }

	int getCols() const { return cols; }

	int getDim() const { return rows * cols; }

	double operator[](int i) const
	{
		if (a_.getRows() != b_.getRows() || a_.getCols() != b_.getCols())
			throw std::out_of_range("a_.getRows() != b_.getRows() || a_.getCols() != b_.getCols()");
		return a_[i] + b_[i];
	}

	double operator()(int i, int j) const { return (*this)[i * getCols() + j]; }
};

template<typename A, typename B>
struct MatrxDifOp 
{
	MatrxDifOp(const A& a, const B& b) : a_(a), b_(b), rows(a_.getRows()), cols(a_.getCols()) {}

	const A& a_;
	const B& b_;

	int rows;
	int cols;

	int getRows() const { return rows; }

	int getCols() const { return cols; }

	int getDim() const { return rows * cols; }

	double operator[](int i) const
	{
		if (a_.getRows() != b_.getRows() || a_.getCols() != b_.getCols())
			throw std::out_of_range("a_.getRows() != b_.getRows() || a_.getCols() != b_.getCols()");
		return a_[i] - b_[i];
	}

	double operator()(int i, int j) const { return (*this)[i * getCols() + j]; }
};

template<typename A, typename B>
struct MatrxMulOp 
{
	MatrxMulOp(const A& a, const B& b) : a_(a), b_(b), rows(a_.getRows()), cols(b_.getCols()) {}

	const A& a_;
	const B& b_;

	int rows;
	int cols;

	int getRows() const { return rows; }

	int getCols() const { return cols; }

	int getDim() const { return rows * cols; }

	double operator()(int i, int j) const
	{
		if (a_.getCols() != b_.getRows())
			throw std::out_of_range("this.rows != other.cols OR this.cols != other.rows");
		double sum = 0.0;
		for (int k = 0; k < a_.getCols(); k++)
			sum += a_(i, k) * b_(k, j);
		return sum;
	}

	double operator[](int idx) const
	{
		int i = idx / cols;
		int j = idx % cols;
		return (*this)(i, j);
	}
};

template<typename A>
struct MatrxDivValOp
{
	MatrxDivValOp(const A& a, const double& b) : a_(a), b_(b), rows(a_.getRows()), cols(a_.getCols()) {}

	const A& a_;
	const double& b_;

	int rows;
	int cols;

	int getRows() const { return rows; }

	int getCols() const { return cols; }

	int getDim() const { return rows * cols; }

	double operator[](int i) const
	{
		return a_[i] / b_;
	}

	double operator()(int i, int j) const { return (*this)[i * getCols() + j]; }
};

template<typename A>
struct MatrxMulValOp
{
	MatrxMulValOp(const A& a, const double& b) : a_(a), b_(b), rows(a_.getRows()), cols(a_.getCols()) {}

	const A& a_;
	const double& b_;

	int rows;
	int cols;

	int getRows() const { return rows; }

	int getCols() const { return cols; }

	int getDim() const { return rows * cols; }

	double operator[](int i) const
	{
		return a_[i] * b_;
	}

	double operator()(int i, int j) const { return (*this)[i * getCols() + j]; }
};


//оператор суммы матриц
template<typename A, typename B>
MatrxSumOp<A, B> operator+ (const A& matrix, const B& other) { return { matrix, other }; }

//оператор разности матриц
template<typename A, typename B>
MatrxDifOp<A, B> operator- (const A& matrix, const B& other) { return { matrix, other }; }

//оператор умножения матриц
template<typename A, typename B>
requires (!std::is_arithmetic_v<B>)
MatrxMulOp<A, B> operator* (const A& matrix, const B& other) { return { matrix, other }; }

//оператор деления матрицы со скаляром
template<typename A>
MatrxDivValOp<A> operator/ (const A& matrix, const double& other) { return { matrix, other }; }

//оператор умножения матрицы со скаляром
template<typename A>
MatrxMulValOp<A> operator* (const A& matrix, const double& other) { return { matrix, other }; }

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

	//конструктор копирования
	//принимает на вход матрицу того же класса
	Matrix(const Matrix& matrix);

	//конструктор выражений
	template<MatrxOp T>
	Matrix(const T& init) : matrix(nullptr), rows(init.getRows()), cols(init.getCols()), dim(init.getDim())
	{
		allocate_memory();
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

	////оператор суммы матриц
	//Matrix operator+ (const Matrix& other) const;

	////оператор разности матриц
	//Matrix operator- (const Matrix& other) const;

	////оператор умножения матриц
	//Matrix operator* (const Matrix& other) const;

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

	//генерация рандомных значений в матрице в пределах от 0 до 1
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

	//очистка матрицы
	void clear();

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
