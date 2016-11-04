#ifndef WAVENET_HIGHPASSOPERATOR_H
#define WAVENET_HIGHPASSOPERATOR_H

/**
 * @file HighpassOperator.h
 * @author Andreas Sogaard
 **/

// STL include(s).
#include <cmath> /* pow */

// Armadillo include(s).
#include <armadillo>

// Wavenet include(s).
#include "Wavenet/MatrixOperator.h"


namespace wavenet {

class HighpassOperator : public MatrixOperator {
    
public:
    
    // Constructor(s).
    HighpassOperator () {};
    
    HighpassOperator (const unsigned& size)
    { setSize(size); };
    
    HighpassOperator (const arma::Col<double>& filter)
    { setFilter(filter); };
    
    HighpassOperator (const arma::Col<double>& filter, const unsigned& size)
    { setSize(size); setFilter(filter); setComplete(true); construct(); };
    
    // Destructor.
    ~HighpassOperator () {};
    
    // Matrix operator methods.
    void setFilter (const arma::Col<double>& filter);
    
};

} // namespace

#endif // WAVENET_HIGHPASSOPERATOR_H
