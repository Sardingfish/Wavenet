#ifndef WAVENET_LOWPASSOPERATOR_H
#define WAVENET_LOWPASSOPERATOR_H

/**
 * @file LowpassOperator.h
 * @author Andreas Sogaard
 **/

// STL include(s).
#include <iostream>
#include <cmath> /* pow */

// ROOT include(s).
// ...

// HepMC include(s).
// ...

// Armadillo include(s).
#include <armadillo>

// Wavenet include(s).
#include "Wavenet/MatrixOperator.h"

using namespace std;
using namespace arma;

class LowpassOperator : public MatrixOperator {
    
public:
    
    // Constructor(s).
    LowpassOperator () {};
    
    LowpassOperator (const unsigned& size)
    { setSize(size); };
    
    LowpassOperator (const arma::Col<double>& filter)
    { setFilter(filter); };
    
    LowpassOperator (const arma::Col<double>& filter, const unsigned& size)
    { setSize(size); setFilter(filter); setComplete(true); construct(); };
    
    // Destructor.
    ~LowpassOperator () {};
    
    // Matrix operator methods.
    void setFilter (const arma::Col<double>& filter);
    
private:
    
};

#endif

