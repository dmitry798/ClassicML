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

	virtual void  train(const string& method, int iters, double lr, int mini_batch, double gamma) = 0;
	virtual Matrix predict() const = 0;
	virtual Matrix predict(Matrix& X_predict) const = 0;
	virtual void loss() const = 0;
};


class LinearRegression: public Models
{
public:

	//конструктор линейной регрессии
	LinearRegression(Dataset& shareData);

	//обучение - Метод Нестерова
	void train(const string& method, int iters = 1000, double lr = 0.01, int mini_batch = 8, double gamma = 0.01) override;

	//тестирование
	Matrix predict() const override;

	//прогноз
	Matrix predict(Matrix& X_predict) const override;

	//ошибка
	void loss() const override;

	~LinearRegression();
};

class LogisticRegression : public Models
{
public:

	//конструктор линейной регрессии
	LogisticRegression(Dataset& shareData);

	//обучение - Метод Нестерова
	void train(const string& method, int iters = 1000, double lr = 0.01, int mini_batch = 8, double gamma = 0.01) override;

	//тестирование
	Matrix predict() const override;

	//прогноз
	Matrix predict(Matrix& X_predict) const override;

	//ошибка
	void loss() const override;

	~LogisticRegression();
};
