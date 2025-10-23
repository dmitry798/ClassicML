#include "../include/ClassicML/models.h"
#include "../include/ClassicML/macros.h"

/************************************************************************************************************************************/

//методы логистической регрессии (+ многоклассовая)

LogisticRegression::LogisticRegression(Dataset& shareData, string way) : Models(shareData), way(way) {}

void LogisticRegression::train(const string& method, int iters, double lr, int mini_batch, double gamma)
{
    if (way == "binary") trainer.choice_train(method, fit, iters, lr, mini_batch, gamma, 2);
    else if (way == "multi") trainer.choice_train(method, fit, iters, lr, mini_batch, gamma, 3);
}

Matrix LogisticRegression::predict()
{
    if (way == "binary") Y_pred = sigmoid(X_test_norm * W);
    else if (way == "multi") Y_pred = softMax(X_test_norm * W);
    return Y_pred;
}

Matrix LogisticRegression::predict(Matrix& X_predict)
{
    if (way == "binary") Y_pred = sigmoid(Models::predict(X_predict));
    else if (way == "multi") Y_pred = softMax(Models::predict(X_predict));
    return Y_pred;
}

void LogisticRegression::loss(double threshold)
{
    if (way == "binary") error.errorsLogClassifier("logloss", threshold);
    else if (way == "multi") error.errorsLogClassifier("loglossMulti", threshold);
}

LogisticRegression::~LogisticRegression() {}


/************************************************************************************************************************************/