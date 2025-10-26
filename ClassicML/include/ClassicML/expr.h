#pragma once
#include <type_traits>

// В режиме биндингов выражения и универсальные операторы отключены
#ifdef CLASSICML_PYBINDINGS
// Ничего не подключаем в режиме Python-модуля.
#else

// Предварительное объявление вашего класса Matrix.
class Matrix;

// Лёгкая обёртка для ленивого умножения
template<class L, class R>
struct MatrxMulOp {
    const L& lhs;
    const R& rhs;
    MatrxMulOp(const L& l, const R& r) : lhs(l), rhs(r) {}
};

// Признак "это Matrix"
template<class T>
inline constexpr bool is_matrix_v = std::is_same_v<std::decay_t<T>, Matrix>;

// Универсальный operator* включается ТОЛЬКО если участвует Matrix хотя бы с одной стороны
template<class L, class R,
    std::enable_if_t<is_matrix_v<L> || is_matrix_v<R>, int> = 0>
inline auto operator*(const L & l, const R & r) {
    return MatrxMulOp<L, R>{l, r};
}

#endif // CLASSICML_PYBINDINGS