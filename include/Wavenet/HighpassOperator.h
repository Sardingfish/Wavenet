#ifndef WAVENET_HIGHPASSOPERATOR_H
#define WAVENET_HIGHPASSOPERATOR_H

/**
 * @file   HighpassOperator.h
 * @author Andreas Sogaard
 * @date   14 November 2016 
 * @brief  Class for high-pass operators used in the wavelet/neural net transform.
 */

// STL include(s).
#include <cmath> /* pow */

// Armadillo include(s).
#include <armadillo>

// Wavenet include(s).
#include "Wavenet/MatrixOperator.h"


namespace wavenet {

/**
 * Class for high-pass operators used in the wavelet/neural net transform.
 * 
 * This derived class implements the mix-in wavenet::MatrixOperator class for 
 * the case of high-pass wavelet operators. In this case, the setFilter method 
 * is non-trivial trivial, since the provided filter is assumed to be composed 
 * of low-pass filter coefficients and therefore the internal filter has to be 
 * computed from the relation
 *   b_{k} = (-1)^k a_{N - k - 1}
 * where {b_{k}} is the set of high-pass filter coefficients, {a_{k}} is the set
 * of low-pass filter coefficients, and N is the number of such coefficients.
 */
class HighpassOperator : public MatrixOperator {
    
public:
    
    /// Constructor(s).
    HighpassOperator () {};
    
    HighpassOperator (const unsigned& size)
    { setSize(size); };
    
    HighpassOperator (const arma::Col<double>& filter)
    { setFilter(filter); };
    
    HighpassOperator (const arma::Col<double>& filter, const unsigned& size)
    { setSize(size); setFilter(filter); setComplete(); construct(); };
    

    /// Destructor.
    ~HighpassOperator () {};


    /// Matrix operator methods.
    // Implementation of virtual method to specify the internal filter from a 
    // set of low-pass filter coefficients
    virtual void setFilter (const arma::Col<double>& filter);
    
};

} // namespace

#endif // WAVENET_HIGHPASSOPERATOR_H
