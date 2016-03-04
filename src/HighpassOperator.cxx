#include "HighpassOperator.h"

void HighpassOperator::setFilter (const arma::Col<double>& filter) {
    unsigned N = filter.n_elem;
    _filter.zeros(N);
    for (unsigned i = 0; i < N; i++) {
        _filter(i) = pow(-1, i) * filter(N - i - 1);
    }
    if (_size) { setComplete(true); }
    return;
}
