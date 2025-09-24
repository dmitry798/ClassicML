#pragma once
#include "matrix_operations.h"
#include "preprocessor.h"
#include "fit.h"

class Models
{
protected:

	Data& data;
	Fit fit;
	Matrix W;

public:

	Models(Data& shareData);

	virtual void train() = 0;
	virtual Matrix predict() const = 0;
	virtual Matrix predict(Matrix& X_predict) const = 0;
};


class LinearRegression: public Models
{
public:

	//����������� �������� ���������
	LinearRegression(Data& shareData);

	//��������
	void train() override;

	//������������
	Matrix predict() const override;

	//"������������"
	Matrix predict(Matrix& X_predict) const override;

	////������
	//double loss() override;

	~LinearRegression();
};


