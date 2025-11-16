#pragma once
#include "matrix.h"
#include "preprocessor.h"
#include "optimization.h"
#include "errors.h"


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
	virtual Matrix predict(Matrix& X_predict);
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
	Matrix predict();

	//прогноз
	Matrix predict(Matrix& X_predict) override;

	//ошибка
	void loss();

	~LinearRegression();
};


//модель логистической регрессии (+ многоклассовая)
class LogisticRegression : public Models
{
public:

	//конструктор логистической регрессии
	LogisticRegression(Dataset& shareData, string way = "binary");

	//обучение
	void train(const string& method, int iters = 1000, double lr = 0.01, int mini_batch = 8, double gamma = 0.01);

	//тестирование
	Matrix predict();

	//прогноз
	Matrix predict(Matrix& X_predict) override;

	//ошибка
	void loss(double threshold);

	~LogisticRegression();
private:

	string way;
};


//модель Knn-классификации (+ взвешанная версия)
class Knn : public Models
{
private:

	int num_neighbors;

	string weighted;

	Distance dist_method;

public:

	//конструктор Knn
	Knn(Dataset& shareData, int num_neighbors, string weighted = "uniform");

	//тестирование
	Matrix predict(string distance = "evklid");

	//прогноз
	Matrix predict(Matrix& X_predict, string distance = "evklid");

	//ошибка
	void loss(double threshold);

	~Knn();
};


//модель Knn-регрессии (+ взвешанная версия)
class KnnRegression : public Models
{
private:

	int num_neighbors;

	string weighted;

	Distance dist_method;

public:

	//конструктор Knn
	KnnRegression(Dataset& shareData, int num_neighbors, string weighted = "uniform");

	//тестирование
	Matrix predict(string distance = "evklid");

	//прогноз
	Matrix predict(Matrix& X_predict, string distance = "evklid");

	//ошибка
	void loss();

	~KnnRegression();
};

class KMeans: public Models
{
private:

	int k;

	int max_iters;

	Matrix centroids;

	Distance dist_method;

public:

	KMeans(Dataset& data, int k, int max_iters);

	//тестирование
	void train(string method = "base", string distance = "evklid");

	//прогноз
	void predict(Matrix& X_predict, string distance = "evklid");

	Matrix getCentroids();

	void loss();

};