#include "matrix_operations.h"

Matrix::Matrix(int* matrix, int lenght, int width)
{
	this->matrix = matrix;
	this->lenght = lenght;
	this->width = width;
}
int Matrix::get_lenght()
{
	return this->lenght;
}

int Matrix::get_width()
{
	return this->width;
}
//Matrix& Matrix::operator+ (const Matrix& other)
//{
//	
//}
//
//Matrix& Matrix::operator- (const Matrix& other) 
//{
//
//}
//
Matrix& Matrix::operator* (const Matrix& other) 
{

}
//
//Matrix& Matrix::operator/ (const Matrix& other) 
//{
//
//}
//
//Matrix& Matrix::transponation(const Matrix& other)
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

}
int main()
{
	Matrix* a = new Matrix(5, 10);

	return 0;
}