#include "matrix_operations.h"
#include "ctime"

Matrix::Matrix(int rows, int cols)
{
	this->rows = rows;
	this->cols = cols;
	this->matrix = new int* [rows];
	for (int i = 0; i < this->rows; i++)
	{
		this->matrix[i] = new int[cols];
	}
}

Matrix::Matrix(int** matrix)
{
	this->matrix = new int*[this->rows];
	for (int i = 0; i < this->rows; i++)
	{
		this->matrix[i] = new int[cols];
		for (int j = 0; j < this->cols; j++)
		{
			this->matrix[i][j] = matrix[i][j];
		}
	}
}

Matrix::Matrix(Matrix& matrix)
{
	this->rows = matrix.rows;
	this->cols = matrix.cols;
	this->matrix = new int* [rows];
	for (int i = 0; i < this->rows; i++) 
	{
		this->matrix[i] = new int[cols];
		for (int j = 0; j < this->cols; j++) 
		{
			this->matrix[i][j] = matrix(i, j);
		}
	}
}

int Matrix::get_rows()
{
	this->rows = sizeof(this->matrix) / sizeof(this->matrix[0]);
	return this->rows;
}

int Matrix::get_cols()
{
	this->cols = sizeof(this->matrix[0]) / sizeof(this->matrix[0][0]);
	return this->cols;
}

int Matrix::operator() (int i, int j)
{
	return matrix[i][j];
}

//Matrix Matrix::operator+ (const Matrix& other)
//{
//	
//}
//
//Matrix Matrix::operator- (const Matrix& other) 
//{
//
//}

Matrix Matrix::operator* (Matrix& other) 
{
	Matrix result(other.get_rows(), other.get_cols());

	for (int i = 0; i < rows; i++)
	{
		result.matrix[i][i] = 0;
		for (int k = 0; k < rows; k++)
		{
			result.matrix[i][k] = 0;
			for (int j = 0; j < cols; j++)
			{
				result.matrix[i][k] += this->matrix[i][j] * other(j, i);
			}
		}
	}
	return result;
}
//
//Matrix Matrix::operator/ (const Matrix& other) 
//{
//
//}
//
//Matrix Matrix::transponation(const Matrix& other)
//{
//
//}
//
//void Matrix::determinant(const Matrix& other)
//{
//
//}
//
//int Matrix::reshape(const Matrix& other)
//{
//
//}

Matrix::~Matrix()
{
	for (int i = 0; i < this->rows; i++)
	{
		delete[] this->matrix[i];
	}
	delete[] matrix;
}


int main()
{
	int** a = new int* [2];
	for (int i = 0; i < 2; i++) 
	{
		a[i] = new int[3] {1, 2, 3};
	}
	int** b = new int* [3];
	for (int i = 0; i < 3; i++)
	{
		b[i] = new int[2] {1 + i, 2 + i};
	}

	Matrix A(a), B(b);

	Matrix C = A * B;

	return 0;
}