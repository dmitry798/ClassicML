#include "../include/ClassicML/models.h"
#include "../include/ClassicML/macros.h"


KMeans::KMeans(Dataset& shareData, int k, int max_iters) : Models(shareData), k(k), max_iters(max_iters), dist_method(shareData), centroids(k, X_train.getCols(), "centroids") {}

void KMeans::train(string method, string rho)
{
    //инициализация центроидов
    srand(time(0));
    if (method == "base")
    {
        for (int c = 0; c < k; c++)
        {
            int choice = rand() % X_train.getRows();
            for (int j = 0; j < X_train_norm.getCols(); j++)
            {
                centroids(c, j) = X_train_norm(choice, j);
            }
        }
    }
    else if(method == "pp")
    {
        int choice = rand() % X_train.getRows();
        for (int j = 0; j < X_train_norm.getCols(); j++)
        {
            centroids(0, j) = X_train_norm(choice, j);
        }
        for (int i = 0; i < k - 1; i++)
        {
            Matrix distances(X_train_norm.getRows(), 1, "distances_centroids_pp");
            for (int j = 0; j < X_train_norm.getRows(); j++)
            {
                Matrix dist(i + 1, 1, "dist_centroid_pp");
                Matrix point = X_train_norm.sliceRow(j, j + 1);

                for (int c = 0; c <= i; c++)
                {
                    Matrix centroid = centroids.sliceRow(c, c + 1);
                    Matrix diff = centroid - point;
                    double distance = 0.0;
                    for (int j = 0; j < diff.getCols(); j++)
                    {
                        distance += diff[j] * diff[j];
                    }
                    distance = sqrt(distance);
                    dist[c] = distance;
                }
                distances[j] = dist[min_idx(dist)];
            }
            int obj_ind = max_idx(distances);
            for (int j = 0; j < X_train_norm.getCols(); j++)
            {
                centroids(i + 1, j) = X_train_norm(obj_ind, j);
            }
        }
    }

    Matrix labels(X_train_norm.getRows(), 1, "labels");

    int i = 0;
    bool converged = true;

    while (i < max_iters && converged)
    {
        converged = false;
        //рассчитываем расстояние
        for (int j = 0; j < X_train_norm.getRows(); j++)
        {
            Matrix dist(k, 1, "distances");
            Matrix point = X_train_norm.sliceRow(j, j + 1);

            for (int c = 0; c < k; c++)
            {
                Matrix centroid = centroids.sliceRow(c, c + 1);
                Matrix diff = centroid - point;
                double distance = 0.0;
                if (rho == "evklid")
                    for (int j = 0; j < diff.getCols(); j++)
                        distance += diff[j] * diff[j];
                else if (rho == "manhattan")
                    for (int j = 0; j < X_train.getCols(); j++)
                        distance += abs(diff[j]);

                distance = sqrt(distance);
                dist[c] = distance;
            }
            int cluster_ind = min_idx(dist);

            labels[j] = cluster_ind;
        }

        Matrix clusters(X_train_norm.getRows(), k);
        for (int i = 0; i < X_train_norm.getRows(); i++)
        {
            int cluster = static_cast<int>(labels[i]);
            clusters(i, cluster) = 1.0;
        }
        Matrix new_centroids(k, X_train.getCols(), "new_centroids");
        Matrix counts(k, 1, "counts");

        for (int i = 0; i < X_train_norm.getRows(); i++)
        {
            for (int c = 0; c < k; c++)
            {
                if (clusters(i, c) == 1.0)
                {
                    counts[c]++;
                    for (int j = 0; j < X_train_norm.getCols(); j++)
                    {
                        new_centroids(c, j) += X_train_norm(i, j);
                    }
                }
            }
        }
        for (int c = 0; c < k; c++)
        {
            if (counts[c] > 0)
            {
                for (int j = 0; j < X_train_norm.getCols(); j++)
                {
                    new_centroids(c, j) /= counts[c];
                }
            }
            else
            {
                int random_idx = rand() % X_train_norm.getRows();
                for (int j = 0; j < X_train_norm.getCols(); j++)
                {
                    new_centroids(c, j) = X_train_norm(random_idx, j);
                }
            }
        }

        double tolerance = 1e-4;
        int c = 0;

        while (c < k && converged)
        {
            int j = 0;
            while (j < centroids.getCols() && converged)
            {
                if (abs(centroids(c, j) - new_centroids(c, j)) > tolerance) converged = false;
                j++;
            }
            c++;
        }

        centroids = new_centroids;
        i++;
    }

    Y_pred = labels;
}

void KMeans::predict(Matrix& X_predict, string rho)
{
    StandardScaler scaler(data);

    Matrix&& X_predict_norm = scaler.normalize(X_predict, mean_x, std_x);


    Y_pred = Matrix(X_predict_norm.getRows(), Y_train.getCols());

    Matrix labels(X_predict_norm.getRows(), 1, "labels");
    for (int j = 0; j < X_predict_norm.getRows(); j++)
    {
        Matrix dist(k, 1, "distances");
        Matrix point = X_predict_norm.sliceRow(j, j + 1);

        for (int c = 0; c < k; c++)
        {
            Matrix centroid = centroids.sliceRow(c, c + 1);
            Matrix diff = centroid - point;
            double distance = 0.0;
            if (rho == "evklid")
                for (int j = 0; j < diff.getCols(); j++)
                    distance += diff[j] * diff[j];
            else if (rho == "manhattan")
                for (int j = 0; j < X_train.getCols(); j++)
                    distance += abs(diff[j]);

            distance = sqrt(distance);
            dist[c] = distance;
        }
        int cluster_ind = min_idx(dist);

        labels[j] = cluster_ind;
    }
    Y_pred = labels;
}

Matrix KMeans::getCentroids()
{
    return centroids;
}

void KMeans::loss()
{
    cout << "Inertia: " << error.inertia(this->getCentroids()) << endl;
}