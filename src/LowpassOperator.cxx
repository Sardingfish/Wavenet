#include "Wavenet/LowpassOperator.h"

namespace wavenet {
    
void LowpassOperator::setFilter (const arma::Col<double>& filter) {

    // Check whether number of filter coefficients is even.
    if (filter.n_elem % 2 != 0) {
        WARNING("Trying to set odd number of filter coefficients (%d).", filter.n_elem);
        return;
    }

    // Set matrix operator filter from set of low-pass filter coefficients.
    m_filter = filter;

    // If the filter completes the matrix operator, label it so.
    if (size()) { setComplete(); }

    return;
}

} // namespace
