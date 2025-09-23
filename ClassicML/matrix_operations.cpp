#include "matrix_operations.h"
#include "ctime"

//реализовать исключения!!!

Matrix::Matrix() : rows(0), cols(0), dim(rows * cols), name(""), matrix(nullptr) {}

Matrix::Matrix(const int rws,const int cls, string nme) : rows(rws), cols(cls), name(nme), dim(rows* cols), matrix(nullptr)
{
	if (dim > 0)
	{
		allocate_memory();
		zeros();
	}
}

Matrix::Matrix(double** matrix, int rws, int cls, string nme) : matrix(nullptr), rows(rws), cols(cls), name(nme), dim(rows* cols)
{
	if (dim > 0)
	{
		allocate_memory();
		copy_data(matrix);
	}
}

Matrix::Matrix(const Matrix& matrix): rows(matrix.rows), cols(matrix.cols), dim(matrix.dim), matrix(nullptr)
{
	if (dim > 0)
	{
		allocate_memory();
		copy_from(matrix);
	}
}

Matrix::Matrix(Matrix&& other) noexcept : rows(other.rows), cols(other.cols), name(other.name), dim(other.dim), matrix(other.matrix)
{
	other.matrix = nullptr;
	other.rows = 0;
	other.cols = 0;
	other.name = "";
}

int Matrix::get_rows() const 
{
	return this->rows;
}

int Matrix::get_cols() const
{
	return this->cols;
}

int Matrix::get_dim() const
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
	int other_cols = other.get_cols();
	if(cols != other.rows)
		throw std::out_of_range("this.rows != other.cols OR this.cols != other.rows");

	Matrix result(rows, other_cols, "res");
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
	Matrix result(rows, cols, "res");

	for (int i = 0; i < dim; i++)
	{
		result.matrix[i] = matrix[i] * value;
	}
	return result;
}

Matrix Matrix::operator+(double value) const
{
	Matrix result(rows, cols);

	for (int i = 0; i < dim; i++)
	{
		result.matrix[i] = matrix[i] + value;
	}
	return result;
}

Matrix Matrix::operator-(double value) const
{
	Matrix result(rows, cols, "res");

	for (int i = 0; i < dim; i++)
	{
		result.matrix[i] = matrix[i] - value;
	}
	return result;
}

Matrix Matrix::operator/(double value) const
{
	Matrix result(rows, cols, "res");

	for (int i = 0; i < dim; i++)
	{
		result.matrix[i] = matrix[i] / value;
	}
	return result;
}

Matrix Matrix::operator+ (const Matrix& other) const
{
	Matrix result(rows, cols, "res");
	if (result.get_dim() == other.dim)
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
	Matrix result(rows, cols, "res");
	if (result.get_dim() == other.dim)
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
		free_memory();
}

//template <int R, int C>
//double** copy_static_memory(double(&matrix)[R][C])
//{
//	double** dest = new double* [R];
//	for (int i = 0; i < R; i++)
//	{
//		dest[i] = new double[C];
//	}
//	for (int i = 0; i < R; i++)
//	{
//		for (int j = 0; j < C; j++)
//		{
//			dest[i][j] = matrix[i][j];
//		}
//	}
//	return dest;
//}
//
//static void free_memory_(double** matrix, int rows)
//{
//	for (int i = 0; i < rows; i++)
//	{
//		delete[] matrix[i];
//	}
//	delete[] matrix;
//	matrix = nullptr;
//}
//
//int main()
//{
//	double a[4][3] = {
//		{1, 2, 3},
//		{4, 5, 6},
//		{7, 8, 9},
//		{10, 11, 12}
//	};
//
//	double c[1][1] = { {1} };
//
//	double f[4][3] = {
//		{1, 2, 3},
//		{4, 5, 6},
//		{7, 8, 9},
//		{10, 11, 12}
//	};
//
//	double b[3][4] = {
//	{1, 2, 3, 4},
//	{5, 6, 7, 8},
//	{9, 10, 11, 12}
//	};
//
//	double** aa = copy_static_memory(a);
//	double** ff = copy_static_memory(f);
//
//	/*double a[][4] = {
//		{1.0, 50.0, 1.0, 1.0},
//		{1.0, 75.0, 2.0, 1.0},
//		{1.0, 100.0, 3.0, 2.0},
//		{1.0, 120.0, 3.0, 3.0},
//		{1.0, 150.0, 4.0, 2.0},
//		{1.0, 80.0, 2.0, 5.0},
//		{1.0, 95.0, 3.0, 1.0},
//		{1.0, 110.0, 3.0, 4.0},
//		{1.0, 130.0, 4.0, 3.0},
//		{1.0, 160.0, 5.0, 2.0}
//	};*/
//
//	Matrix A(aa, 4, 3);
//	Matrix B(ff, 4, 3);
//	Matrix C;
//	C = (A - B * 2).transpons();
//	A.print();
//	B.print();
//	C.print();
//
//	free_memory_(aa, 4);
//	free_memory_(ff, 3);
//
//	return 0;
//}