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
	Optimizer fit;
	Errors error;

public:

	Models(Dataset& shareData);

	virtual void trn() = 0;
	virtual Matrix predict() const = 0;
	virtual Matrix predict(Matrix& X_predict) const = 0;
};


class LinearRegression: public Models
{
public:

	//конструктор линейной регрессии
	LinearRegression(Dataset& shareData);

	//обучение - Сингулярное разложение
	void trn() override;

	//обучение - Метод Нестерова
	void train(int iters, double learning_rate, double partion_save_grade);

	//тестирование
	Matrix predict() const override;

	//прогноз
	Matrix predict(Matrix& X_predict) const override;

	//ошибка
	void loss();

	~LinearRegression();
};


