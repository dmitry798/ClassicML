#include "errors.h"
#include <cmath>
#include "macros.h"

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
    Matrix predictions = Y_pred;
    Matrix residuals = Y_test - predictions;

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