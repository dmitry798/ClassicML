//#include "../include/ClassicML.h"
//#define CLASSICML_PYBINDINGS 1
//#include <pybind11/pybind11.h>
//#include <pybind11/stl.h>
//
//namespace py = pybind11;
//
//PYBIND11_MODULE(classicml, m) {
//    m.doc() = "ClassicML C++ bindings";
//
//    py::class_<Matrix>(m, "Matrix")
//        .def(py::init<int, int>(), py::arg("rows"), py::arg("cols"))
//        .def("getRows", &Matrix::getRows)
//        .def("getCols", &Matrix::getCols)
//        .def("getDim", &Matrix::getDim)
//        .def("print", &Matrix::print)
//        .def("transpose", &Matrix::transpose)
//        .def("sum", &Matrix::sum);
//
//    py::class_<Dataset>(m, "Dataset")
//        .def(py::init<Matrix&, Matrix&>())
//        .def("info", &Dataset::info);
//
//    py::class_<StandardScaler>(m, "StandardScaler")
//        .def(py::init<Dataset&>())
//        .def("split", &StandardScaler::split)
//        .def("standartNormalize", &StandardScaler::standartNormalize);
//
//    py::class_<LinearRegression>(m, "LinearRegression")
//        .def(py::init<Dataset&>())
//        .def("train", &LinearRegression::train,
//            py::arg("method"),
//            py::arg("iters") = 1000,
//            py::arg("lr") = 0.01,
//            py::arg("mini_batch") = 8,
//            py::arg("gamma") = 0.01)
//        .def("predict", py::overload_cast<>(&LinearRegression::predict))
//        .def("predict", py::overload_cast<Matrix&>(&LinearRegression::predict))
//        .def("loss", &LinearRegression::loss);
//
//    py::class_<LogisticRegression>(m, "LogisticRegression")
//        .def(py::init<Dataset&, std::string>(),
//            py::arg("shareData"),
//            py::arg("way") = "binary")
//        .def("train", &LogisticRegression::train,
//            py::arg("method"),
//            py::arg("iters") = 1000,
//            py::arg("lr") = 0.01,
//            py::arg("mini_batch") = 8,
//            py::arg("gamma") = 0.01)
//        .def("predict", py::overload_cast<>(&LogisticRegression::predict))
//        .def("predict", py::overload_cast<Matrix&>(&LogisticRegression::predict))
//        .def("loss", &LogisticRegression::loss, py::arg("threshold") = 0.5);
//
//    py::class_<Knn>(m, "KnnClassifier")
//        .def(py::init<Dataset&, int, std::string>(),
//            py::arg("shareData"),
//            py::arg("num_neighbors"),
//            py::arg("weighted") = "uniform")
//        .def("predict", py::overload_cast<std::string>(&Knn::predict),
//            py::arg("distance") = "evklid")
//        .def("predict", py::overload_cast<Matrix&, std::string>(&Knn::predict),
//            py::arg("X_predict"),
//            py::arg("distance") = "evklid")
//        .def("loss", &Knn::loss, py::arg("threshold") = 0.5);
//
//    // Добавьте привязки для других классов по необходимости
//    py::class_<KnnRegression>(m, "KnnRegression")
//        .def(py::init<Dataset&, int, std::string>(),
//            py::arg("shareData"),
//            py::arg("num_neighbors"),
//            py::arg("weighted") = "uniform")
//        .def("predict", py::overload_cast<std::string>(&KnnRegression::predict),
//            py::arg("distance") = "evklid")
//        .def("predict", py::overload_cast<Matrix&, std::string>(&KnnRegression::predict),
//            py::arg("X_predict"),
//            py::arg("distance") = "evklid")
//        .def("loss", &KnnRegression::loss);
//}