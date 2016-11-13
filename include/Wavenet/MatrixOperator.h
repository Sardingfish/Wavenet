#ifndef WAVENET_MATRIXOPERATOR_H
#define WAVENET_MATRIXOPERATOR_H

/**
 * @file MatrixOperator.h
 * @author Andreas Sogaard
 */

// STL include(s).
#include <cmath> /* pow */
#include <cstdlib> /* abs */
#include <algorithm> /* std::rotate */

// Armadillo include(s).
#include <armadillo>

// Wavenet include(s).
#include "Wavenet/Utilities.h"
#include "Wavenet/Logger.h"


namespace wavenet {

/**
 * Abstract class for matrix operators used in the wavelet/neural net transform.
 *
 * This base class handles the construction of the matrix operator (inheriting
 * from arma::Mat) given a set of filter coefficients. The methods for 
 * specifying the filter coefficients in purely virtual and need to be 
 * implemented by derived classes (low- and high-pass operators).
 */
class MatrixOperator : public arma::Mat<double>, public Logger {
    
public:
    
    /// Constructor(s).
    MatrixOperator () {};
    
    /// Destructor.
    ~MatrixOperator () {};
    
    /// Set method(s).
    inline void setSize (const unsigned& size) {
        _size = size;
        if (_filter.n_elem) { setComplete(true); }
        return;
    }
    
    inline void setComplete (const bool& complete = true) {
        _complete = complete;
        return;
    }
    
    /// Get method(s).
    inline unsigned size () { return _size; }
    inline bool complete () { return _complete; }

    /// Matrix operator methods.
    // Purely virtual method for setting the filter coefficients correctly. 
    virtual void setFilter (const arma::Col<double>& filter) = 0;
    
    
public:
    
    void construct (); // Main method
    void _constructByRows ();
    void _constructByIndices ();
    
    
protected:
    
    // Data member(s).
    unsigned _size;
    arma::Col<double> _filter;
    bool _complete = false;
    
};

/// Rowshift utility functions, for constructing the matrix operator.
// Inplace shift the input arma row right by 'shift' steps.
void rowshift (arma::Row<double>& row, const int& shift);

// Rowshift, given an arma matrix subview as input.
inline arma::Row<double> rowshift(const arma::subview_row<double>& row, const int& shift) {
    arma::Row<double> newRow = row;
    rowshift(newRow, shift);
    return newRow;
}

} // namespace

#endif // WAVENET_MATRIXOPERATOR_H
