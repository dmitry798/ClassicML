#pragma once

#include <iostream>
using std::cout;
using std::endl;
using std::move;
using std::string;

#include <type_traits>
#include <stdexcept>

//сумма матриц
template<typename A, typename B>
struct MatrxSumOp {
    MatrxSumOp(const A& a, const B& b)
        : a_(a), b_(b), rows(a_.getRows()), cols(a_.getCols()) {
    }

    const A& a_;
    const B& b_;
    int rows;
    int cols;

    int getRows() const { return rows; }
    int getCols() const { return cols; }
    int getDim()  const { return rows * cols; }

    double operator[](int i) const {
        if (a_.getRows() != b_.getRows() || a_.getCols() != b_.getCols())
            throw std::out_of_range("a_.getRows() != b_.getRows() || a_.getCols() != b_.getCols()");
        return a_[i] + b_[i];
    }

    double operator()(int i, int j) const { return (*this)[i * getCols() + j]; }
};

//разность матриц
template<typename A, typename B>
struct MatrxDifOp {
    MatrxDifOp(const A& a, const B& b)
        : a_(a), b_(b), rows(a_.getRows()), cols(a_.getCols()) {
    }

    const A& a_;
    const B& b_;
    int rows;
    int cols;

    int getRows() const { return rows; }
    int getCols() const { return cols; }
    int getDim()  const { return rows * cols; }

    double operator[](int i) const {
        if (a_.getRows() != b_.getRows() || a_.getCols() != b_.getCols())
            throw std::out_of_range("a_.getRows() != b_.getRows() || a_.getCols() != b_.getCols()");
        return a_[i] - b_[i];
    }

    double operator()(int i, int j) const { return (*this)[i * getCols() + j]; }
};

// УМНОЖЕНИЕ МАТРИЦ
template<typename A, typename B>
struct MatrxMulOp {
    MatrxMulOp(const A& a, const B& b)
        : a_(a), b_(b), rows(a_.getRows()), cols(b_.getCols()) {
    }

    const A& a_;
    const B& b_;
    int rows;
    int cols;

    int getRows() const { return rows; }
    int getCols() const { return cols; }
    int getDim()  const { return rows * cols; }

    double operator()(int i, int j) const {
        if (a_.getCols() != b_.getRows())
            throw std::out_of_range("this.rows != other.cols OR this.cols != other.rows");
        double sum = 0.0;
        for (int k = 0; k < a_.getCols(); ++k)
            sum += a_(i, k) * b_(k, j);
        return sum;
    }

    double operator[](int idx) const {
        int i = idx / cols;
        int j = idx % cols;
        return (*this)(i, j);
    }
};

// ПОЭЛЕМЕНТНОЕ УМНОЖЕНИЕ МАТРИЦ
template<typename A, typename B>
struct MatrxMulElOp {
    MatrxMulElOp(const A& a, const B& b)
        : a_(a), b_(b), rows(a_.getRows()), cols(b_.getCols()) {
    }

    const A& a_;
    const B& b_;
    int rows;
    int cols;

    int getRows() const { return rows; }
    int getCols() const { return cols; }
    int getDim()  const { return rows * cols; }

    double operator[](int i) const {
        if (a_.getRows() != b_.getRows() || a_.getCols() != b_.getCols())
            throw std::out_of_range("a_.getRows() != b_.getRows() || a_.getCols() != b_.getCols()");
        return a_[i] * b_[i];
    }

    double operator()(int i, int j) const { return (*this)[i * getCols() + j]; }
};

// ДЕЛЕНИЕ МАТРИЦЫ НА СКАЛЯР (A / val)
template<typename A>
struct MatrxDivValOp {
    MatrxDivValOp(const A& a, const double& b)
        : a_(a), b_(b), rows(a_.getRows()), cols(a_.getCols()) {
    }

    const A& a_;
    const double& b_;
    int rows;
    int cols;

    int getRows() const { return rows; }
    int getCols() const { return cols; }
    int getDim()  const { return rows * cols; }

    double operator[](int i) const {
        double bv = b_;
        if (bv <= 1e-6) bv = 1e-6;
        return a_[i] / bv;
    }

    double operator()(int i, int j) const { return (*this)[i * getCols() + j]; }
};

// ДЕЛЕНИЕ СКАЛЯРА НА МАТРИЦУ (val / A)
template<typename A>
struct MatrxDivVal2Op {
    MatrxDivVal2Op(const double& b, const A& a)
        : a_(a), b_(b), rows(a_.getRows()), cols(a_.getCols()) {
    }

    const A& a_;
    const double& b_;
    int rows;
    int cols;

    int getRows() const { return rows; }
    int getCols() const { return cols; }
    int getDim()  const { return rows * cols; }

    double operator[](int i) const {
        double v = a_[i];
        if (v <= 1e-6) v = 1e-6;
        return b_ / v;
    }

    double operator()(int i, int j) const { return (*this)[i * getCols() + j]; }
};

// УМНОЖЕНИЕ МАТРИЦЫ НА СКАЛЯР
template<typename A>
struct MatrxMulValOp {
    MatrxMulValOp(const A& a, const double& b)
        : a_(a), b_(b), rows(a_.getRows()), cols(a_.getCols()) {
    }

    const A& a_;
    const double& b_;
    int rows;
    int cols;

    int getRows() const { return rows; }
    int getCols() const { return cols; }
    int getDim()  const { return rows * cols; }

    double operator[](int i) const { return a_[i] * b_; }

    double operator()(int i, int j) const { return (*this)[i * getCols() + j]; }
};

// РАЗНОСТЬ МАТРИЦЫ И СКАЛЯРА (A - val)
template<typename A>
struct MatrxDifValOp {
    MatrxDifValOp(const A& a, const double& b)
        : a_(a), b_(b), rows(a_.getRows()), cols(a_.getCols()) {
    }

    const A& a_;
    const double& b_;
    int rows;
    int cols;

    int getRows() const { return rows; }
    int getCols() const { return cols; }
    int getDim()  const { return rows * cols; }

    double operator[](int i) const { return a_[i] - b_; }

    double operator()(int i, int j) const { return (*this)[i * getCols() + j]; }
};

// РАЗНОСТЬ СКАЛЯРА И МАТРИЦЫ (val - A)
template<typename A>
struct MatrxDifValOp2 {
    MatrxDifValOp2(const double& b, const A& a)
        : a_(a), b_(b), rows(a_.getRows()), cols(a_.getCols()) {
    }

    const A& a_;
    const double& b_;
    int rows;
    int cols;

    int getRows() const { return rows; }
    int getCols() const { return cols; }
    int getDim()  const { return rows * cols; }

    double operator[](int i) const { return b_ - a_[i]; }

    double operator()(int i, int j) const { return (*this)[i * getCols() + j]; }
};

// СУММА МАТРИЦЫ И СКАЛЯРА
template<typename A>
struct MatrxSumValOp {
    MatrxSumValOp(const A& a, const double& b)
        : a_(a), b_(b), rows(a_.getRows()), cols(a_.getCols()) {
    }

    const A& a_;
    const double& b_;
    int rows;
    int cols;

    int getRows() const { return rows; }
    int getCols() const { return cols; }
    int getDim()  const { return rows * cols; }

    double operator[](int i) const { return a_[i] + b_; }

    double operator()(int i, int j) const { return (*this)[i * getCols() + j]; }
};

//---------------------- ПЕРЕГРУЗКИ ОПЕРАТОРОВ ----------------------//

// Матрица-Матрица: отключаем, если RHS арифметический тип
template<typename A, typename B>
typename std::enable_if<!std::is_arithmetic<B>::value, MatrxSumOp<A, B>>::type
operator+(const A& matrix, const B& other) {
    return MatrxSumOp<A, B>(matrix, other);
}

template<typename A, typename B>
typename std::enable_if<!std::is_arithmetic<B>::value, MatrxDifOp<A, B>>::type
operator-(const A& matrix, const B& other) {
    return MatrxDifOp<A, B>(matrix, other);
}

template<typename A, typename B>
typename std::enable_if<!std::is_arithmetic<B>::value, MatrxMulOp<A, B>>::type
operator*(const A& matrix, const B& other) {
    return MatrxMulOp<A, B>(matrix, other);
}

template<typename A, typename B>
typename std::enable_if<!std::is_arithmetic<B>::value, MatrxMulElOp<A, B>>::type
operator&(const A& matrix, const B& other) {
    return MatrxMulElOp<A, B>(matrix, other);
}

// Матрица-Скаляр
template<typename A>
MatrxDivValOp<A> operator/(const A& matrix, const double& other) {
    return MatrxDivValOp<A>(matrix, other);
}

template<typename A>
MatrxDivVal2Op<A> operator/(const double& other, const A& matrix) {
    return MatrxDivVal2Op<A>(other, matrix);
}

template<typename A>
MatrxMulValOp<A> operator*(const A& matrix, const double& other) {
    return MatrxMulValOp<A>(matrix, other);
}

template<typename A>
MatrxDifValOp<A> operator-(const A& matrix, const double& other) {
    return MatrxDifValOp<A>(matrix, other);
}

template<typename A>
MatrxDifValOp2<A> operator-(const double& other, const A& matrix) {
    return MatrxDifValOp2<A>(other, matrix);
}

template<typename A>
MatrxSumValOp<A> operator+(const A& matrix, const double& other) {
    return MatrxSumValOp<A>(matrix, other);
}
