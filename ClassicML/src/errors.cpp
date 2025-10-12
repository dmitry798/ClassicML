#include "../include/ClassicML/errors.h"
#include <cmath>
#include "../include/ClassicML/macros.h"

Errors::Errors(Dataset& data): data(data) {}

double Errors::MSE() const
{
	Matrix er = Y_pred - Y_test;
	er = er.transpose() * er / er.getDim();
	return er[0];
}

double Errors::RMSE() const
{
    return sqrt(MSE());
}

double Errors::MAE() const
{
	Matrix er = Y_pred - Y_test;
	double sum = 0.0;
	for (int i = 0; i < er.getDim(); i++) 
		sum += abs(er[i]);
	return sum / er.getDim();
}

double Errors::R2() const
{
    Matrix residuals = Y_test - Y_pred;

    double sum_sq_errors = 0.0;
    for (int i = 0; i < residuals.getDim(); i++)
        sum_sq_errors += residuals[i] * residuals[i];

    mean_y.mean(Y_test);

    double sum_sq_total = 0.0;
    for (int i = 0; i < Y_test.getCols(); i++)
    {
        for (int j = 0; j < Y_test.getRows(); j++)
        {
            double diff = Y_test(j, i) - mean_y[i];
            sum_sq_total += diff * diff;
        }
    }

    return 1.0 - (sum_sq_errors / sum_sq_total);
}

void Errors::errorsRegression() const
{
    cout <<
        "MSE: " << MSE() << ", " << endl <<
        "RMSE: " << RMSE() << ", " << endl <<
        "MAE: " << MAE() << ", " << endl <<
        "R2: " << R2()
        << endl;
}

double Errors::logLoss() const
{
    //          logloss = -1/n * sum(y_i * log(p_i) + (1-y_i)*log(1-p_i)) -> min
    double eps = 1e-5;
    Matrix P = sigmoid(X_train * W);
    P.print("P = "); // Посмотреть вероятности

    Matrix logP = Matrix(P + eps).logMatrx();
    logP.print("log(P+eps) = ");

    Matrix logOneMinusP = Matrix(1.0 - P + eps).logMatrx();
    logOneMinusP.print("log(1-P+eps) = ");

    Matrix term1 = Y_train & logP;
    Matrix term2 = (1.0 - Y_train) & logOneMinusP;
    term1.print("Y * logP = ");
    term2.print("(1-Y) * log(1-P) = ");

    Matrix sumMat = term1 + term2;
    sumMat.print("sumMat = ");

    double logloss = -1.0 / Y_train.getDim() * sumMat.sum();
    cout << "logloss = " << logloss << endl;
    return logloss;
}