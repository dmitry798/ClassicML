#include "../ClassicML/include/ClassicML/models.h"
#include "../ClassicML/include/ClassicML/macros.h"

/************************************************************************************************************************************/

//методы логистической регрессии

LogisticRegression::LogisticRegression(Dataset& shareData, string way) : Models(shareData), way(way), model(shareData) {}

void LogisticRegression::train(const string& method, int iters, double lr, int mini_batch, double gamma)
{
    if (way == "binary")
    {
        trainer.choice_train(method, fit, iters, lr, mini_batch, gamma);
    }
    else if (way == "multi")
    {
        model.train(method, iters, lr, mini_batch, gamma);
    }
}

Matrix LogisticRegression::predict()
{
    if (way == "binary")
    {
        Y_pred = sigmoid(X_test_norm * W);
        return Y_pred;
    }
    else if (way == "multi")
    {
        return model.predict();
    }
}

Matrix LogisticRegression::predict(Matrix& X_predict)
{
    if (way == "binary")
    {
        Y_pred = sigmoid(Models::predict(X_predict));
        return Y_pred;
    }
    else if (way == "multi")
    {
        return model.predict(X_predict);
    }
}

void LogisticRegression::loss(double threshold)
{
    if (way == "binary")
        error.errorsLogClassifier("logloss", threshold);
    else if (way == "multi")
        model.loss(threshold);
}

LogisticRegression::~LogisticRegression() {}


/************************************************************************************************************************************/


//методы мультиклассовой логистической регрессии

LogisticRegression::MultiClassLogisticRegression::MultiClassLogisticRegression(Dataset& shareData) : Models(shareData) {}

void LogisticRegression::MultiClassLogisticRegression::train(const string& method, int iters, double lr, int mini_batch, double gamma)
{
    trainer.choice_train(method, fit, iters, lr, mini_batch, gamma);
}

Matrix LogisticRegression::MultiClassLogisticRegression::predict()
{
    Y_pred = softMax(X_test_norm * W);
    return Y_pred;
}

Matrix LogisticRegression::MultiClassLogisticRegression::predict(Matrix& X_predict)
{
    Y_pred = softMax(Models::predict(X_predict));
    return Y_pred;
}

void LogisticRegression::MultiClassLogisticRegression::loss(double threshold)
{
    error.errorsLogClassifier("loglossMulti", threshold);
}

LogisticRegression::MultiClassLogisticRegression::~MultiClassLogisticRegression() {}

/************************************************************************************************************************************/