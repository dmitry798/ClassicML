#include "../include/ClassicML/errors.h"
#include <cmath>

Errors::Errors(Dataset& data): data(data) {}

double Errors::MSE() const
{
	Matrix er = data.Y_pred - data.Y_test;
	er = er.transpose() * er / er.getDim();
	return er[0];
}

double Errors::RMSE() const
{
    return sqrt(MSE());
}

double Errors::MAE() const
{
	Matrix er = data.Y_pred - data.Y_test;
	double sum = 0.0;
	for (int i = 0; i < er.getDim(); i++) 
		sum += abs(er[i]);
	return sum / er.getDim();
}

double Errors::R2() const
{
    Matrix residuals = data.Y_test - data.Y_pred;

    double sum_sq_errors = 0.0;
    for (int i = 0; i < residuals.getDim(); i++)
        sum_sq_errors += residuals[i] * residuals[i];

    data.mean_y = mean(data.Y_test);

    double sum_sq_total = 0.0;
    for (int i = 0; i < data.Y_test.getCols(); i++)
    {
        for (int j = 0; j < data.Y_test.getRows(); j++)
        {
            double diff = data.Y_test(j, i) - data.mean_y[i];
            sum_sq_total += diff * diff;
        }
    }

    return 1.0 - (sum_sq_errors / sum_sq_total);
}

void Errors::errorsRegression() const
{
    cout <<
        "MSE: " << MSE() << endl <<
        "RMSE: " << RMSE() << endl <<
        "MAE: " << MAE() << endl <<
        "R2: " << R2() << endl
        << endl;
}

double Errors::logLoss() const
{
    //          logloss = -1/n * sum(y_i * log(p_i) + (1-y_i)*log(1-p_i)) -> min
    double eps = 1e-15;
    double logloss = (-1.0 / data.Y_train.getRows()) * (Matrix((data.Y_train & Matrix(sigmoid(data.X_train_norm * data.W) + eps).logMatrx()) +
        ((1.0 - data.Y_train) & Matrix(1.0 - sigmoid(data.X_train_norm * data.W) + eps).logMatrx()))).sum();

    return logloss;
}

double Errors::logLossMulti() const
{
    //          logloss = -1/n * sum(sum(y_i * log(p_i)))
    double eps = 1e-15;
    double logloss = (-1.0 / data.Y_train.getRows()) * Matrix(data.Y_train & Matrix(softMax(data.X_train_norm * data.W) + eps).logMatrx()).sum();
    return logloss;
}

double Errors::accuracy(double threshold) const
{
    double true_answer = data.Y_pred.getRows();
    double pred_answer = 0.0;

    for (int i = 0; i < data.Y_pred.getDim(); i++)
    {
        if (data.Y_pred[i] >= threshold && data.Y_test[i] == 1)
            pred_answer++;
    }
    return pred_answer / true_answer;
}

double Errors::accuracyMultiClss() const
{
    double true_answer = data.Y_pred.getRows();
    double pred_answer = 0.0;

    for (int i = 0; i < data.Y_pred.getDim(); i++)
    {
        if (data.Y_pred[i] == data.Y_test[i])
            pred_answer++;
    }
    return pred_answer / true_answer;
}

double Errors::precision(double threshold) const
{
    int tp = 0;
    int fp = 0;

    for (int i = 0; i < data.Y_pred.getDim(); i++)
    {
        if (data.Y_pred[i] >= threshold && data.Y_test[i] == 1) tp++;
        else if (data.Y_pred[i] >= threshold && data.Y_test[i] == 0 ) fp++;
    }
    if (tp + fp > 0) return static_cast<double>(tp) / static_cast<double>(tp + fp);
    else return 0;
}

double Errors::recall(double threshold) const
{
    int tp = 0;
    int fn = 0;

    for (int i = 0; i < data.Y_pred.getDim(); i++)
    {
        if (data.Y_pred[i] >= threshold && data.Y_test[i] == 1) tp++;
        else if (data.Y_pred[i] < threshold && data.Y_test[i] == 1) fn++;
    }
    if (tp + fn > 0) return static_cast<double>(tp) / static_cast<double>(tp + fn);
    else return 0;
}

double Errors::f1Score(double threshold) const
{
    double result = precision(threshold) + recall(threshold);
    if (result > 0)
        return (2.0 * precision(threshold) * recall(threshold)) / result;
    else
        return 0;
}

void Errors::errorsLogClassifier(string name, double threshold) const
{
    if (name == "logloss")
        cout <<
        "logLoss: " << logLoss() << endl;
    else if (name == "loglossMulti")
        cout <<
        "logLoss: " << logLossMulti() << endl;
    cout <<
        "accuracy: " << accuracy(threshold) << endl <<
        "precision: " << precision(threshold) << endl <<
        "recall: " << recall(threshold) << endl <<
        "f1Score: " << f1Score(threshold) << endl
        << endl;

}

void Errors::errorsKnnClassifier(double threshold) const
{
    cout <<
        "accuracy: " << accuracyMultiClss() << endl <<
        "precision: " << precision(threshold) << endl <<
        "recall: " << recall(threshold) << endl <<
        "f1Score: " << f1Score(threshold) << endl
        << endl;
}

double Errors::inertia(Matrix centroids) const
{
    double total_inertia = 0.0;

    for (int i = 0; i < data.X_train_norm.getRows(); i++)
    {
        Matrix point = data.X_train_norm.sliceRow(i, i + 1);
        int cluster = static_cast<int>(data.Y_pred[i]);

        Matrix centroid = centroids.sliceRow(cluster, cluster + 1);
        Matrix diff = centroid - point;

        double distance_sq = 0.0;
        for (int j = 0; j < diff.getCols(); j++)
        {
            distance_sq += diff[j] * diff[j];
        }

        total_inertia += distance_sq;
    }

    return total_inertia;
}