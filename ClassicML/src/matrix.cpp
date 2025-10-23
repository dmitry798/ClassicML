#include "../include/ClassicML/matrix.h"
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
		allocateMemory();
		zeros();
		//cout << "Const1 " << name << endl;
	}
}

Matrix::Matrix(double** matrix, int rws, int cls, string nme) : matrix(nullptr), rows(rws), cols(cls), name(nme), dim(rows* cols)
{
	if (dim > 0)
	{
		allocateMemory();
		copyData(matrix);
		//cout << "Const2 " << name << endl;
	}
}

Matrix::Matrix(double* matrix, int rws, int cls, string nme) : matrix(nullptr), rows(rws), cols(cls), name(nme), dim(rows* cols)
{
	if (dim > 0)
	{
		allocateMemory();
		copyVector(matrix);
		//cout << "Const3 " << name << endl;
	}
}

Matrix::Matrix(const Matrix& matrix): rows(matrix.rows), cols(matrix.cols), dim(matrix.dim), matrix(nullptr)
{
	if (dim > 0)
	{
		allocateMemory();
		copyFrom(matrix);
		//cout << "Const-copy " << name << endl;
	}
}

Matrix::Matrix(Matrix&& other) noexcept : rows(other.rows), cols(other.cols), name(other.name), dim(other.dim), matrix(other.matrix)
{
	other.matrix = nullptr;
	other.rows = 0;
	other.cols = 0;
	other.name = "";
	other.dim = 0;
	//cout << "Const-move " << name << endl;
}

int Matrix::getRows() const { return this->rows; }

int Matrix::getCols() const { return this->cols; }

int Matrix::getDim() const { return this->dim; }

double& Matrix::operator() (int i, int j) const
{
	if (i >= rows || j >= cols || i < 0 || j < 0)
		throw std::out_of_range("Matrix index out of bounds");
	return matrix[i * cols + j];
}

double& Matrix::operator[] (const int i) const
{
	if (i >= dim || i < 0)
		throw std::out_of_range("Matrix index out of bounds");
	return matrix[i];
}

Matrix& Matrix::operator= (Matrix&& other) noexcept
{
	if (this != &other) {
		freeMemory();
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
				freeMemory();
				rows = other.rows;
				cols = other.cols;
				dim = other.dim;
				allocateMemory();
			}
			else
			{
				rows = other.rows;
				cols = other.cols;
				dim = other.dim;
			}
		}
		copyFrom(other);
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
		matrix[i] = (rand() % 1000 - 500) / 10000.0;;
	}
}

void Matrix::randomShuffle(Matrix& other)
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

double Matrix::lenVec()
{
	double len = 0.0;
	for (int i = 0; i < dim; i++)
	{
		len += matrix[i] * matrix[i];
	}
	return sqrt(len);

}

Matrix Matrix::sliceRow(int start, int end)
{
	if (start >= rows || end >= rows || start < 0 || end < 0 || end < start)
		std::out_of_range("start >= dim || end >= dim || start < 0 || end < 0 || end < start");

	int new_rows = end - start;
	Matrix result(new_rows, cols, "res_slice");

	for (int i = 0; i < new_rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			result.matrix[i * cols + j] = matrix[(i + start) * cols + j];
		}
	}

	return result;
}

Matrix Matrix::sliceCols(int start, int end)
{
	if (start >= cols || end >= cols || start < 0 || end < 0 || end < start)
		std::out_of_range("start >= dim || end >= dim || start < 0 || end < 0 || end < start");

	int new_cols = end - start;
	Matrix result(rows, new_cols, "res_slice");

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < new_cols; j++)
		{
			result.matrix[i * new_cols + j] = matrix[i * cols + (j + start)];
		}
	}

	return result;
}

Matrix Matrix::logMatrx()
{
	for (int i = 0; i < dim; i++)
		matrix[i] = log(matrix[i]);
	return *this;
}

double Matrix::sum()
{
	double sum_el = 0.0;
	for (int i = 0; i < dim; i++)
		sum_el += matrix[i];
	return sum_el;
}

Matrix sort(Matrix matrix)
{

	for (int i = 0; i < matrix.getDim(); i++)
	{
		for (int j = 0; j < matrix.getRows(); j++)
		{
			if (matrix[i] < matrix[j])
			{
				double temp = matrix[i];
				matrix[i] = matrix[j];
				matrix[j] = temp;
			}
		}
	}

	return matrix;
}

int mode(const Matrix& col)
{
	int n = col.getRows();
	int maxCount = 0;
	int modeValue = 0;

	for (int i = 0; i < n; i++)
	{
		int count = 0;
		double current = col[i];

		for (int j = 0; j < n; j++)
		{
			if (col[j] == current)
				count++;
		}

		if (count > maxCount)
		{
			maxCount = count;
			modeValue = static_cast<int>(current);
		}
	}

	return modeValue;
}

Matrix Matrix::unique()
{
	double* unique = new double[dim];
	int count_elements = 0;
	for (int i = 0; i < dim; i++) 
	{
		bool check = false;
		int j = 0;
		while (j < count_elements && check == false)
		{
			if (matrix[i] == unique[j])
			{ 
				check = true;
			}
			j++;
		}
		if (check == false)
		{ 
			unique[count_elements] = matrix[i]; 
			count_elements++;
		}
	}

	Matrix result(unique, count_elements, 1, "unique");
	delete[] unique;
	return result;
}
Matrix Matrix::sortRows(int t)
{
	for (int i = 0; i < this->getRows() - 1; i++)
	{
		for (int j = 0; j < this->getRows() - i - 1; j++)
		{
			if ((*this)(j, t) > (*this)(j + 1, t))
			{
				for (int k = 0; k < this->getCols(); k++)
				{
					double temp = (*this)(j, k);
					(*this)(j, k) = (*this)(j + 1, k);
					(*this)(j + 1, k) = temp;
				}
			}
		}
	}
	return *this;
}
//
//Matrix Matrix::roundMatrx()
//{
//	Matrix result(rows, cols, "res-round");
//	for (int i = 0; i < dim; i++)
//	{
//		result[i] = round(matrix[i]);
//	}
//	return result;
//}
//
//Matrix Matrix::sqrtMatrx()
//{
//	Matrix result(rows, cols, "res-sqrt");
//	for (int i = 0; i < dim; i++)
//	{
//		result[i] = sqrt(result[i]);
//	}
//	return result;
//}
//
//Matrix Matrix::absMatrx()
//{
//	Matrix result(rows, cols, "res-abs");
//	for (int i = 0; i < dim; i++)
//	{
//		result[i] = abs(result[i]);
//	}
//	return result;
//}

void Matrix::allocateMemory()
{
	if (matrix == nullptr && dim > 0)
	{
		matrix = new double[dim];
	}
}

void Matrix::freeMemory()
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

void Matrix::copyFrom(const Matrix& other)
{
	if (dim > 0)
	{
		for (int i = 0; i < dim; i++)
		{
			this->matrix[i] = other.matrix[i];
		}
	}
}

void Matrix::copyData(double** matrix)
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

void Matrix::copyVector(double* matrix)
{
	if (dim > 0)
	{
		for (int i = 0; i < dim; i++)
		{
			this->matrix[i] = matrix[i];
		}
	}
}

void Matrix::print(string text) const
{
	cout << text << endl;
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
		freeMemory();
}

