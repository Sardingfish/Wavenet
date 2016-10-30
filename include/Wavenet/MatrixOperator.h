#ifndef WAVENET_MATRIXOPERATOR_H
#define WAVENET_MATRIXOPERATOR_H

/**
 * @file MatrixOperator.h
 * @author Andreas Sogaard
 **/

// STL include(s).
#include <iostream>
#include <cmath> /* pow */
#include <cstdlib> /* abs */
#include <algorithm> /* for_each, rotate */

// ROOT include(s).
// ...

// HepMC include(s).
// ...


// Armadillo include(s).
#include <armadillo>

// Wavenet include(s).
#include "Wavenet/Utils.h"

using namespace std;
using namespace arma;


class MatrixOperator : public arma::Mat<double> {
    
public:
    
    // Constructor(s).
    MatrixOperator () {};
    
    // Destructor.
    ~MatrixOperator () {};
    
    // Set methods(s).
    inline void setSize (const unsigned& size) {
        _size = size;
        if (_filter.n_elem) { setComplete(true); }
        return;
    }
    
    inline void setComplete (const bool& complete = true) {
        _complete = complete;
        return;
    }
    
    // Get methods(s).
    inline unsigned size () { return _size; }
    inline bool complete () { return _complete; }
    
    
protected:
    
    void construct ();
    
    
protected:
    
    // Data member(s).
    unsigned _size;
    arma::Col<double> _filter;
    bool _complete = false;
    
};


// Rowshift, given an arma row vector as input.
void rowshift (arma::Row<double>& row, const int& shift);

// Rowshift, given an arma matrix subview as input.
inline arma::Row<double> rowshift(arma::subview_row<double> row, const int& shift) {
    arma::Row<double> newRow = row;
    rowshift(newRow, shift);
    return newRow;
}

#endif

