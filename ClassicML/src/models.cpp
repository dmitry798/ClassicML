#include <iostream>
#include "../include/ClassicML/models.h"
#include "../include/ClassicML/macros.h"

/************************************************************************************************************************************/

//выбор оптимизаторов

Models::Trainer::Trainer() {}

void Models::Trainer::choice_train(const string& method, Optimizer& fit, int iters, double lr, int mini_batch, double gamma, int num_method)
{
    if (method == "nesterov") fit.sgdNesterov(iters, lr, mini_batch, gamma, num_method);
    else if (method == "sgd") fit.sgd(iters, lr, mini_batch, num_method);
    else if (method == "gd") fit.gradientDescent(iters, lr, num_method);
    else if (method == "momentum") fit.sgdMomentum(iters, lr, mini_batch, gamma, num_method);
    else throw std::runtime_error("Unknown training method: " + method);
}

/************************************************************************************************************************************/

//общая структура модели

Models::Models(Dataset& shareData) : data(shareData), fit(data), error(shareData) {}

Matrix Models::predict(Matrix& X_predict)
{
    StandardScaler scaler(data);

    Matrix mean_(X_predict.getCols(), 1, "mean"); Matrix std_(X_predict.getCols(), 1, "std");
    mean_ = mean(X_predict); std_ = stddev(X_predict, mean_);

    Matrix&& X_predict_norm = scaler.normalize(X_predict, mean_, std_);

    return X_predict_norm * W;
}

