#pragma once

#include <iostream>
using std::cout;
using std::endl;
using std::move;
using std::string;

//концепт шаблонов для матричных выражений
template<typename A>
concept MatrxOp = requires(const A & a, int i, int j)
{
	{ a[i] } -> std::same_as<double>;
	{ a(i, j) } -> std::same_as<double>;
	{ a.getRows() } -> std::same_as<int>;
	{ a.getCols() } -> std::same_as<int>;
	{ a.getDim() } -> std::same_as<int>;
};

//структура для суммы матриц
template<typename A, typename B>
struct MatrxSumOp
{
	MatrxSumOp(const A& a, const B& b) : a_(a), b_(b), rows(a_.getRows()), cols(a_.getCols()) {}

	const A& a_;
	const B& b_;

	int rows;
	int cols;

	//получение кол-ва строк
	int getRows() const { return rows; }

	//получение кол-ва колонок
	int getCols() const { return cols; }

	//получение размерности матрицы
	int getDim() const { return rows * cols; }

	//операция суммы матриц
	double operator[](int i) const
	{
		if (a_.getRows() != b_.getRows() || a_.getCols() != b_.getCols())
			throw std::out_of_range("a_.getRows() != b_.getRows() || a_.getCols() != b_.getCols()");
		return a_[i] + b_[i];
	}

	//костыль для шаблонов
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

	//операци разности матриц
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

	//операция умножения матриц
	double operator()(int i, int j) const
	{
		if (a_.getCols() != b_.getRows())
			throw std::out_of_range("this.rows != other.cols OR this.cols != other.rows");
		double sum = 0.0;
		for (int k = 0; k < a_.getCols(); k++)
			sum += a_(i, k) * b_(k, j);
		return sum;
	}

	//костыль для обращения к элементам
	double operator[](int idx) const
	{
		int i = idx / cols;
		int j = idx % cols;
		return (*this)(i, j);
	}
};

template<typename A, typename B>
struct MatrxMulElOp
{
	MatrxMulElOp(const A& a, const B& b) : a_(a), b_(b), rows(a_.getRows()), cols(b_.getCols()) {}

	const A& a_;
	const B& b_;

	int rows;
	int cols;

	int getRows() const { return rows; }

	int getCols() const { return cols; }

	int getDim() const { return rows * cols; }

	//операци поэлементного умножения матриц
	double operator[](int i) const
	{
		if (a_.getRows() != b_.getRows() || a_.getCols() != b_.getCols())
			throw std::out_of_range("a_.getRows() != b_.getRows() || a_.getCols() != b_.getCols()");
		return a_[i] * b_[i];
	}

	double operator()(int i, int j) const { return (*this)[i * getCols() + j]; }
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

	//операция деления матрицы на скаляр
	double operator[](int i) const
	{
		return a_[i] / b_;
	}

	double operator()(int i, int j) const { return (*this)[i * getCols() + j]; }
};

template<typename A>
struct MatrxDivVal2Op
{
	MatrxDivVal2Op(const double& b, const A& a) : a_(a), b_(b), rows(a_.getRows()), cols(a_.getCols()) {}

	const A& a_;
	const double& b_;

	int rows;
	int cols;

	int getRows() const { return rows; }

	int getCols() const { return cols; }

	int getDim() const { return rows * cols; }

	//операция деления матрицы на скаляр
	double operator[](int i) const
	{

		return b_ / a_[i];
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

	//операция умножения матрицы на скаляр
	double operator[](int i) const
	{
		return a_[i] * b_;
	}

	double operator()(int i, int j) const { return (*this)[i * getCols() + j]; }
};

template<typename A>
struct MatrxDifValOp
{
	MatrxDifValOp(const A& a, const double& b) : a_(a), b_(b), rows(a_.getRows()), cols(a_.getCols()) {}

	const A& a_;
	const double& b_;

	int rows;
	int cols;

	int getRows() const { return rows; }

	int getCols() const { return cols; }

	int getDim() const { return rows * cols; }

	//операция деления матрицы на скаляр
	double operator[](int i) const
	{
		return a_[i] - b_;
	}

	double operator()(int i, int j) const { return (*this)[i * getCols() + j]; }
};

template<typename A>
struct MatrxDifValOp2
{
	MatrxDifValOp2(const double& b, const A& a) : a_(a), b_(b), rows(a_.getRows()), cols(a_.getCols()) {}

	const A& a_;
	const double& b_;

	int rows;
	int cols;

	int getRows() const { return rows; }

	int getCols() const { return cols; }

	int getDim() const { return rows * cols; }

	//операция деления матрицы на скаляр
	double operator[](int i) const
	{
		return  b_ - a_[i];
	}

	double operator()(int i, int j) const { return (*this)[i * getCols() + j]; }
};

template<typename A>
struct MatrxSumValOp
{
	MatrxSumValOp(const A& a, const double& b) : a_(a), b_(b), rows(a_.getRows()), cols(a_.getCols()) {}

	const A& a_;
	const double& b_;

	int rows;
	int cols;

	int getRows() const { return rows; }

	int getCols() const { return cols; }

	int getDim() const { return rows * cols; }

	//операция умножения матрицы на скаляр
	double operator[](int i) const
	{
		return a_[i] + b_;
	}

	double operator()(int i, int j) const { return (*this)[i * getCols() + j]; }
};


//оператор суммы матриц
template<typename A, typename B>
	requires (!std::is_arithmetic_v<B>)
MatrxSumOp<A, B> operator+ (const A& matrix, const B& other) { return { matrix, other }; }

//оператор разности матриц
template<typename A, typename B>
	requires (!std::is_arithmetic_v<B>)
MatrxDifOp<A, B> operator- (const A& matrix, const B& other) { return { matrix, other }; }

//оператор умножения матриц
template<typename A, typename B>
	requires (!std::is_arithmetic_v<B>)
MatrxMulOp<A, B> operator* (const A& matrix, const B& other) { return { matrix, other }; }

//оператор поэлементного умножения матриц
template<typename A, typename B>
	requires (!std::is_arithmetic_v<B>)
MatrxMulElOp<A, B> operator& (const A& matrix, const B& other) { return { matrix, other }; }

//оператор деления матрицы со скаляром
template<typename A>
MatrxDivValOp<A> operator/ (const A& matrix, const double& other) { return { matrix, other }; }

//оператор деления матрицы со скаляром
template<typename A>
MatrxDivVal2Op<A> operator/ (const double& other, const A& matrix) { return { other, matrix }; }

//оператор умножения матрицы со скаляром
template<typename A>
MatrxMulValOp<A> operator* (const A& matrix, const double& other) { return { matrix, other }; }

//оператор разности матрицы со скаляром
template<typename A>
MatrxDifValOp<A> operator- (const A& matrix, const double& other) { return { matrix, other }; }

//оператор разности матрицы со скаляром
template<typename A>
MatrxDifValOp2<A> operator- (const double& other, const A& matrix) { return { other, matrix }; }

//оператор суммы матрицы со скаляром
template<typename A>
MatrxSumValOp<A> operator+ (const A& matrix, const double& other) { return { matrix, other }; }