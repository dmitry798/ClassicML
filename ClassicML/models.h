#pragma once
#include "matrix.h"
#include "preprocessor.h"
#include "optimization.h"
#include "errors.h"
using namespace Data;

class Models
{
protected:

	Dataset& data;
	Fit fit;
	Errors error;

public:

	Models(Dataset& shareData);

	virtual void train() = 0;
	virtual Matrix predict() const = 0;
	virtual Matrix predict(Matrix& X_predict) const = 0;
};


class LinearRegression: public Models
{
public:

	//конструктор линейной регрессии
	LinearRegression(Dataset& shareData);

	//обучение
	void train() override;

	//тестирование
	Matrix predict() const override;

	//прогноз
	Matrix predict(Matrix& X_predict) const override;

	//ошибка
	void loss();

	~LinearRegression();
};


