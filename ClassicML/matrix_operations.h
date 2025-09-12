#pragma once
#include <iostream>
using std::cin;
using std::cout;

class Matrix
{
private:

	int rows, cols;

	int** matrix;

public:

	Matrix(int rows, int cols);

	Matrix(int** matrix);

	Matrix(Matrix& matrix);

	int get_rows();

	int get_cols();

	int operator() (int i, int j);

	Matrix operator+= (Matrix& other);

	Matrix operator+ (const Matrix& other);

	Matrix operator- (const Matrix& other);

	Matrix operator* (Matrix& other);

	Matrix operator/ (const Matrix& other);

	Matrix transponation(const Matrix& matrix);

	void determinant(const Matrix& matrix);

	int reshape(const Matrix& matrix);

	~Matrix();

};