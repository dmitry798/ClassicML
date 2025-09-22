#pragma once
#include <iostream>
#include <cmath>
using std::cin; 
using std::cout; 
using std::endl;
using std::move;
using std::string;

class Matrix
{
private:

	int rows, cols, dim;

	string name;

	double* matrix;

public:

	//����������� �� ���������
	Matrix();

	//����������� 1
	Matrix(const int rows, const int cols, string name = "");

	//����������� 2
	Matrix(double** matrix, int rows, int cols, string name = "");

	//����������� �����������
	Matrix(const Matrix& matrix);

	//����������� �����������
	Matrix(Matrix&& other) noexcept;

	//��������� ���-�� �����
	int get_rows() const;

	//��������� ���-�� �������
	int get_cols() const;

	//��������� �����������
	int get_dim() const;

	//�������� ()
	double& operator() (const int i, const int j) const;

	//�������� ����� ������
	Matrix operator+ (const Matrix& other) const;

	//�������� �������� ������
	Matrix operator- (const Matrix& other) const;

	//�������� ��������� ������
	Matrix operator* (const Matrix& other) const;

	//�������� ����� �� ��������
	Matrix operator+ (double value) const;

	//�������� ��������� �������
	Matrix operator- (double value) const;

	//�������� ��������� �� ������
	Matrix operator* (double value) const;

	//�������� ������� �� ������
	Matrix operator/ (double value) const;

	//�������� ������������
	Matrix& operator= (const Matrix& other) noexcept;

	//�������� �����������
	Matrix& operator= (Matrix&& other) noexcept;

	//����������������
	Matrix transpose() const;

	//��������� ��������� �������� � �������
	void random();

	//������������� ����� � �������
	void random_shuffle(Matrix& other);

	//����� ����������� �������
	void reshape() const;

	//����� �������
	void print() const;

	//��� ��������
	Matrix& mean(const Matrix& x);

	//����������� ����������
	Matrix& std(const Matrix& x, const Matrix& mean);

	//����� �������
	double len();

	//����������
	~Matrix();

private:
	
	//��������� ������
	void allocate_memory();

	//������������ ������
	void free_memory();

	//����������� ������� � �����
	void copy_data(double** matrix);

	//����������� ������ ������ ������
	void copy_from(const Matrix& other);

	//���������� ������� ������
	void zeros();
};
