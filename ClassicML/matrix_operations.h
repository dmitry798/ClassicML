#pragma once
#include <iostream>
using std::cin;
using std::cout;

class Matrix
{
private:

	int lenght, width;

	int* matrix;

public:

	Matrix(int* matrix, int lenght, int width);

	int get_lenght();

	int get_width();

	Matrix& operator+ (const Matrix& other);

	Matrix& operator- (const Matrix& other);

	Matrix& operator* (const Matrix& other);

	Matrix& operator/ (const Matrix& other);

	Matrix& transponation(const Matrix& matrix);

	void determinant(const Matrix& matrix);

	int reshape(const Matrix& matrix);

	~Matrix();

};