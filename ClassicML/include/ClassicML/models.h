#pragma once
#include "matrix.h"
#include "preprocessor.h"
#include "optimization.h"
#include "errors.h"
using namespace Data;


//абстрактный класс для всех моделей

class Models
{
protected:

	Dataset& data;
	Optimizer fit;
	Errors error;

	struct Trainer
	{
		Trainer();

		void choice_train(const string& method, Optimizer& fit, int iters = 1000, double lr = 0.01, int mini_batch = 8, double gamma = 0.01, int num_method = 0);
	};
	Trainer trainer;

public:

	Models(Dataset& shareData);

	virtual Matrix predict() const = 0;
	virtual Matrix predict(Matrix& X_predict) const;
};


//модель линейной регрессии

class LinearRegression: public Models
{
public:

	//конструктор линейной регрессии
	LinearRegression(Dataset& shareData);

	//обучение
	void train(const string& method, int iters = 1000, double lr = 0.01, int mini_batch = 8, double gamma = 0.01);

	//тестирование
	Matrix predict() const override;

	//прогноз
	Matrix predict(Matrix& X_predict) const override;

	//ошибка
	void loss() const;

	~LinearRegression();
};


// модель логистической регрессии (+ многоклассовая)

class LogisticRegression : public Models
{
public:

	//конструктор логистической регрессии
	LogisticRegression(Dataset& shareData, string way = "binary");

	//обучение
	void train(const string& method, int iters = 1000, double lr = 0.01, int mini_batch = 8, double gamma = 0.01);

	//тестирование
	Matrix predict() const override;

	//прогноз
	Matrix predict(Matrix& X_predict) const override;

	//ошибка
	void loss(double threshold) const;

	~LogisticRegression();
private:

	string way;


	//многоклассовая логистическая регрессия

	class MultiClassLogisticRegression : public Models
	{
	public:

		//конструктор мультиклассовой логистической регрессии
		MultiClassLogisticRegression(Dataset& shareData);

		//обучение
		void train(const string& method, int iters = 1000, double lr = 0.01, int mini_batch = 8, double gamma = 0.01);

		//тестирование
		Matrix predict() const override;

		//прогноз
		Matrix predict(Matrix& X_predict) const override;

		//ошибка
		void loss(double threshold) const;

		~MultiClassLogisticRegression();
	};

	MultiClassLogisticRegression model;
};



class Knn : public Models
{
public:

	//конструктор Knn
	Knn(Dataset& shareData);

	//обучение
	void train();

	//тестирование
	Matrix predict() const override;

	//прогноз
	Matrix predict(Matrix& X_predict) const override;

	//ошибка
	void loss() const;

	~Knn();
};