#include "Wavenet/HighpassOperator.h"

namespace wavenet {
    
void HighpassOperator::setFilter (const arma::Col<double>& filter) {
    
    // Check whether number of filter coefficients is even.
    if (filter.n_elem % 2 != 0) {
        WARNING("Trying to set odd number of filter coefficients (%d).", filter.n_elem);
        return;
    }

    // Set matrix operator filter from set of low-pass filter coefficients.
    const unsigned N = filter.n_elem;
    m_filter.zeros(N);
    for (unsigned i = 0; i < N; i++) {
        m_filter(i) = pow(-1, i) * filter(N - i - 1);
    }

    // If the filter completes the matrix operator, label it so.
    if (size()) { setComplete(); }

    return;
}

} // namespace
