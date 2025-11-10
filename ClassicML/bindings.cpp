#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

// All ClassicML headers
#include <ClassicML/matrix.h>
#include <ClassicML/preprocessor.h>
#include <ClassicML/errors.h>
#include <ClassicML/optimization.h>
#include <ClassicML/models.h>

namespace py = pybind11;

PYBIND11_MODULE(_classicml, m) {
    m.doc() = "ClassicML - Machine Learning from scratch in C++";

    // ========== MATRIX CLASS ==========
    py::class_<Matrix>(m, "Matrix", py::buffer_protocol())
        // Constructors - ТОЛЬКО ТРИ
        .def(py::init<>(), "Default constructor")
        .def(py::init<int, int, std::string>(),
            py::arg("rows"), py::arg("cols"), py::arg("name") = "",
            "Create matrix with specified dimensions\n\n"
            "Parameters:\n"
            "  rows: Number of rows\n"
            "  cols: Number of columns\n"
            "  name: Optional name for the matrix")
        .def(py::init<const Matrix&>(), "Copy constructor")

        // Buffer protocol for NumPy integration
        .def_buffer([](Matrix& mat) -> py::buffer_info {
        return py::buffer_info(
            nullptr,
            sizeof(double),
            py::format_descriptor<double>::format(),
            2,
            { mat.getRows(), mat.getCols() },
            { sizeof(double) * mat.getCols(), sizeof(double) }
        );
            })

        // Properties
        .def("get_rows", &Matrix::getRows, "Get number of rows")
        .def("get_cols", &Matrix::getCols, "Get number of columns")
        .def("get_dim", &Matrix::getDim, "Get total number of elements")
        .def_property_readonly("shape", [](const Matrix& mat) {
        return std::make_tuple(mat.getRows(), mat.getCols());
            }, "Get matrix shape as (rows, cols)")

        // Indexing operators
        .def("__call__", [](const Matrix& mat, int i, int j) -> double {
        return mat(i, j);
            }, py::arg("i"), py::arg("j"), "Access element at (i, j)")
        .def("__getitem__", [](const Matrix& mat, int i) -> double {
        return mat[i];
            }, py::arg("i"), "Access element by linear index")
        .def("__setitem__", [](Matrix& mat, int i, double val) {
        mat[i] = val;
            }, py::arg("i"), py::arg("value"), "Set element by linear index")
        .def("__len__", [](const Matrix& mat) { return mat.getDim(); })

        // Matrix operations - ИСПОЛЬЗУЕМ ЛЯМБДЫ ДЛЯ MSVC
        .def("transpose", [](Matrix& mat) { return mat.transpose(); },
            "Transpose the matrix")
        .def("slice_row", [](Matrix& mat, int start, int end) {
        return mat.sliceRow(start, end);
            }, py::arg("start"), py::arg("end"), "Slice rows from start to end")
        .def("slice_cols", [](Matrix& mat, int start, int end) {
        return mat.sliceCols(start, end);
            }, py::arg("start"), py::arg("end"), "Slice columns from start to end")

        .def("clear", &Matrix::clear, "Clear all elements to zero")
        .def("zeros", &Matrix::zeros, "Set all elements to zero")
        .def("random", &Matrix::random, "Fill with random values")
        .def("random_shuffle", &Matrix::randomShuffle, py::arg("other"),
            "Randomly shuffle rows along with another matrix")
        .def("sum", &Matrix::sum, "Sum all elements")
        .def("unique", &Matrix::unique, "Get unique elements")
        .def("sort_rows", &Matrix::sortRows, py::arg("column"),
            "Sort rows by specified column")
        .def("log_matrx", &Matrix::logMatrx, "Apply natural logarithm element-wise")
        .def("len_vec", &Matrix::lenVec, "Calculate L2 norm")

        // Utility methods
        .def("print", &Matrix::print, py::arg("text") = "",
            "Print matrix with optional text label")
        .def("reshape", &Matrix::reshape, "Display matrix dimensions")
        .def("copy_from", &Matrix::copyFrom, py::arg("other"),
            "Copy data from another matrix")

        // Representation
        .def("__repr__", [](const Matrix& mat) {
        return "<Matrix shape=(" + std::to_string(mat.getRows()) +
            ", " + std::to_string(mat.getCols()) + ")>";
            });

    // ========== MATRIX HELPER FUNCTIONS ==========
    m.def("mean", [](const Matrix& matrix) { return mean(matrix); },
        py::arg("matrix"), "Calculate mean of each column\n\n"
        "Parameters:\n"
        "  matrix: Input matrix\n"
        "Returns:\n"
        "  Column vector of means");

    m.def("stddev", [](const Matrix& matrix, const Matrix& mean_val) {
        return stddev(matrix, mean_val);
        }, py::arg("matrix"), py::arg("mean"),
            "Calculate standard deviation of each column\n\n"
            "Parameters:\n"
            "  matrix: Input matrix\n"
            "  mean: Mean values for each column\n"
            "Returns:\n"
            "  Column vector of standard deviations");

    m.def("mode", [](const Matrix& column) { return mode(column); },
        py::arg("column"), "Find mode (most frequent value) in a column");

    m.def("sort", [](Matrix& matrix) { return sort(matrix); },
        py::arg("matrix"), "Sort all elements of the matrix");

    // ========== DATASET CLASS ==========
    py::class_<Dataset>(m, "Dataset")
        .def(py::init<Matrix&, Matrix&>(),
            py::arg("X"), py::arg("Y"),
            "Create dataset from feature matrix X and target matrix Y\n\n"
            "Parameters:\n"
            "  X: Feature matrix (n_samples, n_features)\n"
            "  Y: Target matrix (n_samples, n_targets)")

        // Public members - expose as properties
        .def_readwrite("X", &Dataset::X, "Feature matrix")
        .def_readwrite("Y", &Dataset::Y, "Target matrix")
        .def_readwrite("X_train", &Dataset::X_train, "Training features")
        .def_readwrite("Y_train", &Dataset::Y_train, "Training targets")
        .def_readwrite("X_test", &Dataset::X_test, "Test features")
        .def_readwrite("Y_test", &Dataset::Y_test, "Test targets")
        .def_readwrite("X_train_norm", &Dataset::X_train_norm, "Normalized training features")
        .def_readwrite("X_test_norm", &Dataset::X_test_norm, "Normalized test features")
        .def_readwrite("Y_train_norm", &Dataset::Y_train_norm, "Normalized training targets")
        .def_readwrite("Y_test_norm", &Dataset::Y_test_norm, "Normalized test targets")
        .def_readwrite("Y_pred", &Dataset::Y_pred, "Predicted values")
        .def_readwrite("W", &Dataset::W, "Model weights")
        .def_readwrite("mean_x", &Dataset::mean_x, "Feature means")
        .def_readwrite("std_x", &Dataset::std_x, "Feature standard deviations")
        .def_readwrite("mean_y", &Dataset::mean_y, "Target means")
        .def_readwrite("std_y", &Dataset::std_y, "Target standard deviations")

        .def("info", &Dataset::info, "Display dataset information");

    // ========== STANDARD SCALER CLASS ==========
    py::class_<StandardScaler>(m, "StandardScaler")
        .def(py::init<Dataset&>(),
            py::arg("data"),
            "Create StandardScaler for dataset preprocessing\n\n"
            "Parameters:\n"
            "  data: Dataset to preprocess")

        .def("split", &StandardScaler::split,
            py::arg("ratio") = 0.8, py::arg("random") = true,
            "Split dataset into training and test sets\n\n"
            "Parameters:\n"
            "  ratio: Fraction of data for training (default: 0.8)\n"
            "  random: Whether to shuffle data before splitting (default: True)")

        .def("standart_normalize", &StandardScaler::standartNormalize,
            "Apply standard normalization (zero mean, unit variance)")

        .def("normalize", [](StandardScaler& ss, Matrix& matrix,
            Matrix& mean, Matrix& std) {
                return ss.normalize(matrix, mean, std);
            }, py::arg("matrix"), py::arg("mean"), py::arg("std"),
                "Normalize matrix using provided mean and std\n\n"
                "Parameters:\n"
                "  matrix: Matrix to normalize\n"
                "  mean: Mean values\n"
                "  std: Standard deviation values\n"
                "Returns:\n"
                "  Normalized matrix")

        .def("denormalize", [](StandardScaler& ss, Matrix& matrix,
            Matrix& mean, Matrix& std) {
                return ss.denormalize(matrix, mean, std);
            }, py::arg("matrix"), py::arg("mean"), py::arg("std"),
                "Denormalize matrix using provided mean and std\n\n"
                "Parameters:\n"
                "  matrix: Matrix to denormalize\n"
                "  mean: Mean values used in normalization\n"
                "  std: Standard deviation values used in normalization\n"
                "Returns:\n"
                "  Denormalized matrix");

    // ========== PREPROCESSING FUNCTIONS ==========
    m.def("one_hot_encoder", [](Matrix& matrix) {
        return OneHotEncoder(matrix);
        }, py::arg("matrix"),
            "Convert categorical matrix to one-hot encoded representation\n\n"
            "Parameters:\n"
            "  matrix: Input matrix with categorical values\n"
            "Returns:\n"
            "  One-hot encoded matrix");

    m.def("decoder_ohe", [](Matrix& matrix) {
        return DecoderOHE(matrix);
        }, py::arg("matrix"),
            "Decode one-hot encoded matrix back to categorical\n\n"
            "Parameters:\n"
            "  matrix: One-hot encoded matrix\n"
            "Returns:\n"
            "  Categorical matrix");

    // ========== ERRORS CLASS ==========
    py::class_<Errors>(m, "Errors")
        .def(py::init<Dataset&>(),
            py::arg("data"),
            "Create Errors object for computing metrics\n\n"
            "Parameters:\n"
            "  data: Dataset containing predictions and true values")

        // Regression metrics
        .def("mse", &Errors::MSE,
            "Calculate Mean Squared Error\n\n"
            "Returns:\n"
            "  MSE value")
        .def("rmse", &Errors::RMSE,
            "Calculate Root Mean Squared Error\n\n"
            "Returns:\n"
            "  RMSE value")
        .def("mae", &Errors::MAE,
            "Calculate Mean Absolute Error\n\n"
            "Returns:\n"
            "  MAE value")
        .def("r2", &Errors::R2,
            "Calculate R-squared (coefficient of determination)\n\n"
            "Returns:\n"
            "  R² value")

        // Classification metrics
        .def("log_loss", &Errors::logLoss,
            "Calculate binary cross-entropy loss\n\n"
            "Returns:\n"
            "  Log loss value")
        .def("log_loss_multi", &Errors::logLossMulti,
            "Calculate multiclass cross-entropy loss\n\n"
            "Returns:\n"
            "  Log loss value")
        .def("accuracy", &Errors::accuracy,
            py::arg("threshold") = 0.5,
            "Calculate classification accuracy\n\n"
            "Parameters:\n"
            "  threshold: Decision threshold for binary classification (default: 0.5)\n"
            "Returns:\n"
            "  Accuracy value")
        .def("accuracy_multi_clss", &Errors::accuracyMultiClss,
            "Calculate multiclass classification accuracy\n\n"
            "Returns:\n"
            "  Accuracy value")
        .def("precision", &Errors::precision,
            py::arg("threshold") = 0.5,
            "Calculate precision\n\n"
            "Parameters:\n"
            "  threshold: Decision threshold (default: 0.5)\n"
            "Returns:\n"
            "  Precision value")
        .def("recall", &Errors::recall,
            py::arg("threshold") = 0.5,
            "Calculate recall\n\n"
            "Parameters:\n"
            "  threshold: Decision threshold (default: 0.5)\n"
            "Returns:\n"
            "  Recall value")
        .def("f1_score", &Errors::f1Score,
            py::arg("threshold") = 0.5,
            "Calculate F1 score\n\n"
            "Parameters:\n"
            "  threshold: Decision threshold (default: 0.5)\n"
            "Returns:\n"
            "  F1 score value")

        // Print methods
        .def("errors_regression", &Errors::errorsRegression,
            "Print all regression metrics")
        .def("errors_log_classifier", &Errors::errorsLogClassifier,
            py::arg("name"), py::arg("threshold") = 0.5,
            "Print classification metrics")
        .def("errors_knn_classifier", &Errors::errorsKnnClassifier,
            "Print KNN classification metrics");

    // ========== ACTIVATION FUNCTIONS ==========
    m.def("sigmoid", [](Matrix matrix) {
        return sigmoid(std::move(matrix));
        }, py::arg("matrix"),
            "Apply sigmoid activation function\n\n"
            "Parameters:\n"
            "  matrix: Input matrix\n"
            "Returns:\n"
            "  Matrix with sigmoid applied element-wise");

    m.def("softmax", [](Matrix matrix) {
        return softMax(std::move(matrix));
        }, py::arg("matrix"),
            "Apply softmax activation function\n\n"
            "Parameters:\n"
            "  matrix: Input matrix\n"
            "Returns:\n"
            "  Matrix with softmax applied row-wise");

    // ========== MODELS BASE CLASS ==========
    py::class_<Models>(m, "Models")
        .def("predict", [](Models& model, Matrix& X) {
        return model.predict(X);
            }, py::arg("X"),
                "Predict on new data\n\n"
                "Parameters:\n"
                "  X: Feature matrix\n"
                "Returns:\n"
                "  Predictions");

    // ========== LINEAR REGRESSION ==========
    py::class_<LinearRegression, Models>(m, "LinearRegression")
        .def(py::init<Dataset&>(),
            py::arg("data"),
            "Create Linear Regression model\n\n"
            "Parameters:\n"
            "  data: Dataset object containing training data")

        .def("train", &LinearRegression::train,
            py::arg("method") = "gd",
            py::arg("iters") = 1000,
            py::arg("lr") = 0.01,
            py::arg("mini_batch") = 32,
            py::arg("gamma") = 0.9,
            "Train linear regression model\n\n"
            "Parameters:\n"
            "  method: Optimization method - 'gd', 'sgd', 'momentum', 'nesterov', 'svd' (default: 'gd')\n"
            "  iters: Number of training iterations (default: 1000)\n"
            "  lr: Learning rate (default: 0.01)\n"
            "  mini_batch: Mini-batch size for SGD variants (default: 32)\n"
            "  gamma: Momentum coefficient (default: 0.9)")

        .def("predict", [](LinearRegression& lr) {
        return lr.predict();
            }, "Predict on test set\n\n"
            "Returns:\n"
                "  Predictions for test data")

        .def("predict", [](LinearRegression& lr, Matrix& X) {
        return lr.predict(X);
            }, py::arg("X"),
                "Predict on new data\n\n"
                "Parameters:\n"
                "  X: Feature matrix\n"
                "Returns:\n"
                "  Predictions")

        .def("loss", &LinearRegression::loss,
            "Calculate and print regression metrics (MSE, RMSE, MAE, R²)");

    // ========== LOGISTIC REGRESSION ==========
    py::class_<LogisticRegression, Models>(m, "LogisticRegression")
        .def(py::init<Dataset&, std::string>(),
            py::arg("data"), py::arg("way") = "binary",
            "Create Logistic Regression model\n\n"
            "Parameters:\n"
            "  data: Dataset object containing training data\n"
            "  way: Classification type - 'binary' or 'multi' (default: 'binary')")

        .def("train", &LogisticRegression::train,
            py::arg("method") = "gd",
            py::arg("iters") = 1000,
            py::arg("lr") = 0.01,
            py::arg("mini_batch") = 32,
            py::arg("gamma") = 0.9,
            "Train logistic regression model\n\n"
            "Parameters:\n"
            "  method: Optimization method - 'gd', 'sgd', 'momentum', 'nesterov' (default: 'gd')\n"
            "  iters: Number of training iterations (default: 1000)\n"
            "  lr: Learning rate (default: 0.01)\n"
            "  mini_batch: Mini-batch size for SGD variants (default: 32)\n"
            "  gamma: Momentum coefficient (default: 0.9)")

        .def("predict", [](LogisticRegression& lr) {
        return lr.predict();
            }, "Predict on test set\n\n"
            "Returns:\n"
                "  Class probabilities for test data")

        .def("predict", [](LogisticRegression& lr, Matrix& X) {
        return lr.predict(X);
            }, py::arg("X"),
                "Predict on new data\n\n"
                "Parameters:\n"
                "  X: Feature matrix\n"
                "Returns:\n"
                "  Class probabilities")

        .def("loss", &LogisticRegression::loss,
            py::arg("threshold") = 0.5,
            "Calculate and print classification metrics\n\n"
            "Parameters:\n"
            "  threshold: Decision threshold (default: 0.5)");

    // ========== KNN CLASSIFIER ==========
    py::class_<Knn, Models>(m, "Knn")
        .def(py::init<Dataset&, int, std::string>(),
            py::arg("data"),
            py::arg("num_neighbors") = 5,
            py::arg("weighted") = "uniform",
            "Create K-Nearest Neighbors classifier\n\n"
            "Parameters:\n"
            "  data: Dataset object containing training data\n"
            "  num_neighbors: Number of neighbors to consider (default: 5)\n"
            "  weighted: Weight type - 'uniform' or 'distance' (default: 'uniform')")

        .def("predict", [](Knn& knn, std::string distance) {
        return knn.predict(distance);
            }, py::arg("distance") = "evklid",
                "Predict on test set\n\n"
                "Parameters:\n"
                "  distance: Distance metric - 'evklid' or 'manhattan' (default: 'evklid')\n"
                "Returns:\n"
                "  Class predictions for test data")

        .def("predict", [](Knn& knn, Matrix& X, std::string distance) {
        return knn.predict(X, distance);
            }, py::arg("X"), py::arg("distance") = "evklid",
                "Predict on new data\n\n"
                "Parameters:\n"
                "  X: Feature matrix\n"
                "  distance: Distance metric - 'evklid' or 'manhattan' (default: 'evklid')\n"
                "Returns:\n"
                "  Class predictions")

        .def("loss", &Knn::loss,
            py::arg("threshold") = 0.5,
            "Calculate and print classification accuracy\n\n"
            "Parameters:\n"
            "  threshold: Not used for KNN but kept for API consistency");

    // ========== KNN REGRESSION ==========
    py::class_<KnnRegression, Models>(m, "KnnRegression")
        .def(py::init<Dataset&, int, std::string>(),
            py::arg("data"),
            py::arg("num_neighbors") = 5,
            py::arg("weighted") = "uniform",
            "Create K-Nearest Neighbors regressor\n\n"
            "Parameters:\n"
            "  data: Dataset object containing training data\n"
            "  num_neighbors: Number of neighbors to consider (default: 5)\n"
            "  weighted: Weight type - 'uniform' or 'distance' (default: 'uniform')")

        .def("predict", [](KnnRegression& kr, std::string distance) {
        return kr.predict(distance);
            }, py::arg("distance") = "evklid",
                "Predict on test set\n\n"
                "Parameters:\n"
                "  distance: Distance metric - 'evklid' or 'manhattan' (default: 'evklid')\n"
                "Returns:\n"
                "  Predictions for test data")

        .def("predict", [](KnnRegression& kr, Matrix& X, std::string distance) {
        return kr.predict(X, distance);
            }, py::arg("X"), py::arg("distance") = "evklid",
                "Predict on new data\n\n"
                "Parameters:\n"
                "  X: Feature matrix\n"
                "  distance: Distance metric - 'evklid' or 'manhattan' (default: 'evklid')\n"
                "Returns:\n"
                "  Predictions")

        .def("loss", &KnnRegression::loss,
            "Calculate and print regression metrics (MSE, RMSE, MAE, R²)");
    // ========== K-MEANS CLUSTERING ==========
    py::class_<KMeans, Models>(m, "KMeans")
        .def(py::init<Dataset&, int, int>(),
            py::arg("data"),
            py::arg("k") = 3,
            py::arg("max_iters") = 100,
            "Create K-Means clustering model\n\n"
            "Parameters:\n"
            "  data: Dataset object containing training data\n"
            "  k: Number of clusters (default: 3)\n"
            "  max_iters: Maximum number of iterations (default: 100)")

        .def("train", &KMeans::train,
            py::arg("method") = "base",
            py::arg("rho") = "evklid",
            "Train K-Means clustering model\n\n"
            "Parameters:\n"
            "  method: Initialization method - 'base' or 'pp' (K-Means++) (default: 'base')\n"
            "  rho: Distance metric - 'evklid' or 'manhattan' (default: 'evklid')")

        .def("predict", [](KMeans& km, Matrix& X, std::string rho) {
        return km.predict(X, rho);
            }, py::arg("X"), py::arg("rho") = "evklid",
                "Predict clusters on new data\n\n"
                "Parameters:\n"
                "  X: Feature matrix\n"
                "  rho: Distance metric - 'evklid' or 'manhattan' (default: 'evklid')\n"
                "Returns:\n"
                "  Cluster labels")

        .def("get_centroids", &KMeans::getCentroids,
            "Get cluster centroids\n\n"
            "Returns:\n"
            "  Matrix of centroids (k x n_features)")

        .def("loss", &KMeans::loss,
            "Calculate and print clustering metrics (Inertia)")

        .def_property_readonly("centroids", &KMeans::getCentroids,
            "Cluster centroids (read-only property)");

    // ========== EXCEPTION HANDLING ==========
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        }
        catch (const std::runtime_error& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
        catch (const std::out_of_range& e) {
            PyErr_SetString(PyExc_IndexError, e.what());
        }
        catch (const std::invalid_argument& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        }
        catch (const std::exception& e) {
            PyErr_SetString(PyExc_Exception, e.what());
        }
        });

    // ========== VERSION INFO ==========
    m.attr("__version__") = "1.0.0";
}
