#include "../include/ClassicML/models.h"
#include "../include/ClassicML/macros.h"

/************************************************************************************************************************************/

//методы линейной регрессии
LinearRegression::LinearRegression(Dataset& shareData) : Models(shareData) {}

void LinearRegression::train(const string& method, int iters, double lr, int mini_batch, double gamma)
{
    if (method == "svd")
    {
        Matrix U, s, VT;
        fit.svd(U, s, VT);
        W = VT.transpose() * s * U.transpose() * Y_train;
    }
    else trainer.choice_train(method, fit, iters, lr, mini_batch, gamma, 1);
}

Matrix LinearRegression::predict()
{
    StandardScaler scaler(data);
    Y_pred = scaler.denormalize(X_test_norm * W, mean_y, std_y);
    return Y_pred;
}

Matrix LinearRegression::predict(Matrix& X_predict)
{
    return Models::predict(X_predict);
}

void LinearRegression::loss()
{
    error.errorsRegression();
}

LinearRegression::~LinearRegression() {}

/************************************************************************************************************************************/