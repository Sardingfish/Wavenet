#include "Wavenet/LowpassOperator.h"


namespace Wavenet {
    
void LowpassOperator::setFilter (const arma::Col<double>& filter) {
    _filter = filter;
    if (_size) { setComplete(true); }
    return;
}

} // namespace
