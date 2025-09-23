#pragma once
#include "matrix_operations.h"

class LinearRegression
{
private:
	Matrix X;
	Matrix Y;
	Matrix W;

	Matrix X_train;
	Matrix Y_train;
	Matrix X_test;
	Matrix Y_test;

	Matrix X_train_norm;
	Matrix Y_train_norm;
	Matrix X_test_norm;
	Matrix Y_test_norm;

	Matrix mean_x;
	Matrix std_x;
	Matrix mean_y;
	Matrix std_y;

	Matrix U; 
	Matrix s; 
	Matrix VT;

public:

	//����������� �������� ���������
	LinearRegression(Matrix x, Matrix y);

	//����������� �����
	//learning_rate - �������� ��������
	//epoch - ���������� ����
	void train(double learning_rate, int epoch);

	//���-����������...
	Matrix SVD();

	//"������������" ������
	Matrix predict();

	//������ ������
	double loss();

	//������������ ������ �� ��������� � ������������� �������
	void normalizer();

	//���������� ������ �� ����� ������� �� ��������� � �������������
	void split(double ratio);

	~LinearRegression();
private:

	//����������� ������� �������� � ���, �������� ������� norma
	void transform(const Matrix& Z_train, const Matrix& Z_test, Matrix& Z_train_norm, Matrix& Z_test_norm, Matrix& mean_z, Matrix& std_z);

	//������������ ������
	Matrix norma(const Matrix& z, const Matrix& mean_z, const Matrix& std_z);

	//�������������� ������ ��� "������������"
	Matrix denorma(const Matrix& z, const Matrix& mean_z, const Matrix& std_z);
};


