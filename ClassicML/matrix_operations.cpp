﻿#include "matrix_operations.h"
#include "ctime"

//реализовать исключения!!!

Matrix::Matrix() : rows(0), cols(0), dim(rows* cols), name(""), matrix(nullptr) 
{ 
	//cout << "Const def" << endl; 
}

Matrix::Matrix(const int rws,const int cls, string nme) : rows(rws), cols(cls), name(nme), dim(rows* cols), matrix(nullptr)
{
	if (dim > 0)
	{
		allocate_memory();
		zeros();
		//cout << "Const1 " << name << endl;
	}
}

Matrix::Matrix(double** matrix, int rws, int cls, string nme) : matrix(nullptr), rows(rws), cols(cls), name(nme), dim(rows* cols)
{
	if (dim > 0)
	{
		allocate_memory();
		copy_data(matrix);
		//cout << "Const2 " << name << endl;
	}
}

Matrix::Matrix(const Matrix& matrix): rows(matrix.rows), cols(matrix.cols), dim(matrix.dim), matrix(nullptr)
{
	if (dim > 0)
	{
		allocate_memory();
		copy_from(matrix);
		//cout << "Const-copy " << name << endl;
	}
}

Matrix::Matrix(Matrix&& other) noexcept : rows(other.rows), cols(other.cols), name(other.name), dim(other.dim), matrix(other.matrix)
{
	other.matrix = nullptr;
	other.rows = 0;
	other.cols = 0;
	other.name = "";
	//cout << "Const-move " << name << endl;
}

int Matrix::getRows() const 
{
	return this->rows;
}

int Matrix::getCols() const
{
	return this->cols;
}

int Matrix::getDim() const
{
	return this->dim;
}

double& Matrix::operator() (int i, int j) const
{
	if (i >= rows || j >= cols || i < 0 || j < 0)
		throw std::out_of_range("Matrix index out of bounds");
	return matrix[i * cols + j];
}

Matrix Matrix::operator* (const Matrix& other)  const
{
	int other_cols = other.cols;
	if(cols != other.rows)
		throw std::out_of_range("this.rows != other.cols OR this.cols != other.rows");

	Matrix result(rows, other_cols, "res*m");
	for (int i = 0; i < rows; i++)
	{
		double* res = result.matrix + i * other_cols;
		for (int j = 0; j < cols; j++)
		{
			const double* oth = other.matrix + j * other_cols;
			double a = matrix[i * cols + j];
			for (int k = 0; k < other_cols; k++)
			{
				res[k] += a * oth[k];
			}
		}
	}
	return result;
}

Matrix Matrix::operator*(double value) const
{
	Matrix result(rows, cols, "res*v");

	for (int i = 0; i < dim; i++)
	{
		result.matrix[i] = matrix[i] * value;
	}
	return result;
}

Matrix Matrix::operator+(double value) const
{
	Matrix result(rows, cols, "res+v");

	for (int i = 0; i < dim; i++)
	{
		result.matrix[i] = matrix[i] + value;
	}
	return result;
}

Matrix Matrix::operator-(double value) const
{
	Matrix result(rows, cols, "res-v");

	for (int i = 0; i < dim; i++)
	{
		result.matrix[i] = matrix[i] - value;
	}
	return result;
}

Matrix Matrix::operator/(double value) const
{
	Matrix result(rows, cols, "res/v");

	for (int i = 0; i < dim; i++)
	{
		result.matrix[i] = matrix[i] / value;
	}
	return result;
}

Matrix Matrix::operator+ (const Matrix& other) const
{
	Matrix result(rows, cols, "res+m");
	if (result.dim == other.dim)
	{
		for (int i = 0; i < dim; i++)
		{
			result.matrix[i] = matrix[i] + other.matrix[i];
		}
	}
	else
		throw std::out_of_range("Matrix index out of bounds");
	return result;
}
 
Matrix Matrix::operator- (const Matrix& other) const
{
	Matrix result(rows, cols, "res-m");
	if (result.dim == other.dim)
	{
		for (int i = 0; i < dim; i++)
		{
			result.matrix[i] = matrix[i] - other.matrix[i];
		}
	}
	else
		throw std::out_of_range("Matrix index out of bounds");
	return result;
}

Matrix& Matrix::operator= (Matrix&& other) noexcept
{
	if (this != &other) {
		free_memory();
		rows = other.rows;
		cols = other.cols;
		dim = other.dim;
		name = other.name;
		matrix = other.matrix;

		other.rows = 0;
		other.cols = 0;
		other.dim = 0;
		other.name = "";
		other.matrix = nullptr;
	}
	return *this;
}

Matrix& Matrix::operator= (const Matrix& other) noexcept
{
	if (this != &other)
	{
		if (rows != other.rows || cols != other.cols)
		{
			if (matrix == nullptr || other.dim > dim)
			{
				free_memory();
				rows = other.rows;
				cols = other.cols;
				dim = other.dim;
				allocate_memory();
			}
			else
			{
				rows = other.rows;
				cols = other.cols;
				dim = other.dim;
			}
		}
		copy_from(other);
	}
	return *this;
}

Matrix& Matrix::mean(const Matrix& x)
{
	for (int i = 0; i < x.cols; i++)
	{
		double variance = 0;
		for (int j = 0; j < x.rows; j++)
		{
			variance += x(j, i);
		}
		matrix[i] = variance / x.rows;
	}
	return *this;
}

Matrix& Matrix::std(const Matrix& x, const Matrix& mean)
{
	for (int i = 0; i < x.cols; i++)
	{
		double variance = 0;
		for (int j = 0; j < x.rows; j++)
		{
			double diff = x(j, i) - mean.matrix[i];
			variance += diff * diff;
		}
		matrix[i] = sqrt(variance / (x.rows - 1));
	}
	return *this;
}

void Matrix::clear()
{
	zeros();
}

void Matrix::random()
{
	for (int i = 0; i < dim; i++)
	{
		matrix[i] = (rand() % 100) / static_cast<double>(100);
	}
}

void Matrix::random_shuffle(Matrix& other)
{
	for (int i = rows - 1; i > 0; i--) 
	{
		int k = rand() % (i + 1);
		for (int j = 0; j < cols; j++) 
		{
			double temp = matrix[i * cols + j]; 
			matrix[i * cols + j] = matrix[k * cols + j];
			matrix[k * cols + j] = temp;
		}
		for (int j = 0; j < other.cols; j++)
		{
			double temp = other.matrix[i * other.cols + j];
			other.matrix[i * other.cols + j] = other.matrix[k * other.cols + j];
			other.matrix[k * other.cols + j] = temp;
		}
	}
}

void Matrix::zeros()
{
	for (int i = 0; i < dim; i++)
	{
		matrix[i] = 0;
	}
}

Matrix Matrix::transpose() const
{
	Matrix result(cols, rows, "res");
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			result.matrix[j * result.cols + i] = matrix[i * cols + j];
		}
	}
    return result;
}

double Matrix::len()
{
	double len = 0.0;
	for (int i = 0; i < dim; i++)
	{
		len += matrix[i] * matrix[i];
	}
	return sqrt(len);

}

void Matrix::allocate_memory()
{
	if (matrix == nullptr && dim > 0)
	{
		matrix = new double[dim];
	}
}

void Matrix::free_memory()
{
	if (matrix != nullptr)
	{
		delete[] matrix;
		matrix = nullptr;
	}
	rows = 0;
	cols = 0;
	dim = 0;
	name = "";
}

void Matrix::copy_from(const Matrix& other)
{
	if (dim > 0)
	{
		for (int i = 0; i < dim; i++)
		{
			this->matrix[i] = other.matrix[i];
		}
	}
}

void Matrix::copy_data(double** matrix)
{
	if (dim > 0)
	{
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				this->matrix[i * cols + j] = matrix[i][j];
			}
		}
	}
}

void Matrix::print() const
{
	for (int i = 0; i < dim; i++)
	{
		cout << matrix[i] << " ";
		if((i + 1) % cols == 0)
			cout << endl;
	}
	cout << endl;
}

void Matrix::reshape() const
{
	if (dim > 0)
		cout << "(" << rows << "x" << cols << ")" << endl;
}

Matrix::~Matrix()
{
	if (matrix != nullptr)
		//cout << "deConst " << name << endl;
		free_memory();
}