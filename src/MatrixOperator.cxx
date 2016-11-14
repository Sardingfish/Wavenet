#include "Wavenet/MatrixOperator.h"

namespace wavenet {
    
void MatrixOperator::construct () {

    // Check whether the matrix operator is properly configured.
    if (!complete()) {
        WARNING("Trying to construct incomplete operator.");
        return;
    }

    // Choose the most efficient (but equivalent) way to construct the matrix 
    // operator, given the size and filter configuration. The numbers are found 
    // empirically, but generally: if we have a large operator and few filter 
    // coefficients, construct by filter indices; if we have a small operatator 
    // and many indices, construct by matrix rows.
    const unsigned N = m_filter.n_elem;
    if ((N ==  2 && m_size >= 2) ||
        (N <=  6 && m_size >= 3) ||
        (N <= 20 && m_size >= 4) ||
        (           m_size >= 5)) {
        constructByIndices_();
    } else {
        constructByRows_();
    }

    return;
}

void MatrixOperator::constructByRows_ () {
    
    // Initialise variables.
    const unsigned nRows = (unsigned) pow(2, m_size);
    const unsigned nCols = (unsigned) pow(2, m_size + 1);
    const unsigned N = m_filter.n_elem;
    
    // Initialise the matrix operator to all zeros.
    this->zeros(nRows, nCols);

    // Initialise single row matrix with the appropriately positioned filter 
    // coefficients.
    arma::Row<double> v (nCols, arma::fill::zeros);
    for (unsigned i = 0; i < N; i++) {
        v( (i - N/2 + 1) % nCols ) += m_filter(N - i - 1);
    }
    // Use the fact that all rows in the matrix operator are identical modulo 
    // even-numbered shifts to construct the remaining rows by shifted versions 
    // of the base row.
    this->row(0) = v;
    for (unsigned i = nRows; i --> 1; ) {

        // Shift current row two places to the right.
        rowshift(v, 2);

        // And set as i'th row in matrix operator.
        this->row(i) = v;        
    }
    
    return;
}

void MatrixOperator::constructByIndices_ () {
    
    // Initialise variables.
    const unsigned nRows = (unsigned) pow(2, m_size);
    const unsigned nCols = (unsigned) pow(2, m_size + 1);
    const unsigned N = m_filter.n_elem;
    
    // Initialise the matrix operator to all zeros.
    this->zeros(nRows, nCols);
    
    // Initialise vectors of row- and column indices.
    arma::Col<arma::uword> rows (nRows, arma::fill::zeros);
    arma::Col<arma::uword> cols (nRows, arma::fill::zeros);

    // Row indices are always 0, 1, ... , nRows.
    for (unsigned irow = 0; irow < nRows; irow++) {
        rows(irow) = irow;
    }

    // Column indices and chosed appropriately for the first filter coefficient.
    for (unsigned icol = 0; icol < nRows; icol++) {
        cols(icol) = (N / 2 + 2 * icol) % nCols;
    }
    
    // List of combined row- and column indices.
    arma::Col<arma::uword> indices; 

    // List containing the location(s) of indices in the first column, which 
    // cannot be shifted to the left, since the indices are of type arma::uword
    // (unsigned).
    arma::Col<arma::uword> idxZero; 

    // Use the fact that the locations of successive filter coefficients is 
    // related by a shift to the left by a single column.
    for (unsigned i = 0; i < N; i++) {

        // Compute combined row- and column indices (single unsigned int).
        indices = cols * nRows + rows; 

        // Add the i'th filter coefficient at these locations.
        (*this)(indices) += m_filter(i);

        // Shift indices to the left by one column, taking care that we don't 
        // bring an unsigned number below zero.
        idxZero = find(cols == 0);
        cols(idxZero) += nCols;
        cols -= 1;
    }
    
    return;
}

void rowshift (arma::Row<double>& row, const int& shift) {
    
    // Initialise length of row vector.
    const unsigned length = row.n_elem;
    
    if (shift > 0 && shift < length) {

        // If shift is in (0, length), perform the shift in place.
        std::rotate(row.begin(), row.begin() + shift, row.end());
        
    } else {

        // If shift is larger than row length, take appropriate modulo and try 
        // again.
        if (length <  abs(shift)) {
            rowshift(row, sign(shift) * (abs(shift) % length));
        }

        // If shift is negative, add one period (row length) and try again.        
        if (shift  <  0) {
            rowshift(row, length + shift);
        }
        
    }

    // Implicitly: If shift == 0, do nothing.
    
    return;
}

} // namespace
