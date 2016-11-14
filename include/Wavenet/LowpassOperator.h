#ifndef WAVENET_LOWPASSOPERATOR_H
#define WAVENET_LOWPASSOPERATOR_H

/**
 * @file   LowpassOperator.h
 * @author Andreas Sogaard
 * @date   14 November 2016
 * @brief  Class for low-pass operators used in the wavelet/neural net transform.
 */

// STL include(s).
#include <cmath> /* pow */

// Armadillo include(s).
#include <armadillo>

// Wavenet include(s).
#include "Wavenet/MatrixOperator.h"


namespace wavenet {

/**
 * Class for low-pass operators used in the wavelet/neural net transform.
 * 
 * This derived class implements the mix-in wavenet::MatrixOperator class for 
 * the case of low-pass wavelet operators. In this, case the setFilter method is
 * trivial, since the provided filter is assumed to be composed of low-pass
 * filter coefficients.
 */
class LowpassOperator : public MatrixOperator {
    
public:
    
    /// Constructor(s).
    LowpassOperator () {};
    
    LowpassOperator (const unsigned& size)
    { setSize(size); };
    
    LowpassOperator (const arma::Col<double>& filter)
    { setFilter(filter); };
    
    LowpassOperator (const arma::Col<double>& filter, const unsigned& size)
    { setSize(size); setFilter(filter); setComplete(); construct(); };
    

    /// Destructor.
    ~LowpassOperator () {};
    

    /// Matrix operator methods.
    // Implementation of virtual method to specify the internal filter from a 
    // set of low-pass filter coefficients
    virtual void setFilter (const arma::Col<double>& filter);
        
};

} // namespace

#endif // WAVENET_LOWPASSOPERATOR_H
