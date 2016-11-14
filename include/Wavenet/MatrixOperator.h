#ifndef WAVENET_MATRIXOPERATOR_H
#define WAVENET_MATRIXOPERATOR_H

/**
 * @file   MatrixOperator.h
 * @author Andreas Sogaard
 * @date   14 November 2016
 * @brief  Mix-in class for matrix operators used in the wavelet/neural net transform.
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
 * Mix-in class for matrix operators used in the wavelet/neural net transform.
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
        m_size = size;
        if (m_filter.n_elem) { setComplete(); }
        return;
    }

    // Purely virtual method for setting the filter coefficients correctly. Must 
    // be implemented by derived classes. 
    virtual void setFilter (const arma::Col<double>& filter) = 0;
    

    /// Get method(s).
    inline unsigned size () const { return m_size; }
    inline bool complete () const { return m_complete; }


    /// Operator construction method(s).
    // Main interface method. Redirects to appropriate, internal implementation 
    // method, deducing which method to use for optimal performance.
    void construct ();


protected:

    /// Internal set method(s).
    // Should be determined by the appropriate set methods, and is therefore not
    // publicly exposed.
    inline void setComplete (const bool& complete = true) {
        m_complete = complete;
        return;
    }


    /// Internal operator construction method(s).
    // Construct matrix operator by iterating rows.
    void constructByRows_ ();

    // Construct matrix operator by iterating filter indices.
    void constructByIndices_ ();
    
    
protected:

    /// Data member(s).
    // The matrix operator is of size N x 2N, where m_size = log2(N). That is, 
    // m_size is the number of rows (in powers of 2) of the matrix operator. 
    unsigned m_size = 0;

    // The set of filter coefficients used to construct the matix operator.
    arma::Col<double> m_filter = {};

    // Whether the internal configuration of the matrix operator is complete, 
    // and therefore whether it is ready to be constructed.
    bool m_complete = false;


};

/// Rowshift utility functions, for constructing the matrix operator.
// Shift the input arma row _in place_ to the right by 'shift' steps.
void rowshift (arma::Row<double>& row, const int& shift);

// Rowshift, given an arma matrix subview as input.
inline arma::Row<double> rowshift(const arma::subview_row<double>& row, const int& shift) {

    // Copy the matrix subview to a new row.
    arma::Row<double> newRow = row;

    // Rowshift in place.
    rowshift(newRow, shift);

    // Return the shifted copy.
    return newRow;
}

} // namespace

#endif // WAVENET_MATRIXOPERATOR_H
