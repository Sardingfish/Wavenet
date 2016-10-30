#ifndef WAVELETML_HighpassOperator
#define WAVELETML_HighpassOperator

/**
 * @file HighpassOperator.h
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

// WaveletML include(s).
#include "Wavenet/MatrixOperator.h"

using namespace std;
using namespace arma;

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
    
private:
    
};

#endif


