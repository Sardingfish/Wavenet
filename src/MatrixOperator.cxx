#include "Wavenet/MatrixOperator.h"

namespace wavenet {
    
void MatrixOperator::construct () {

    if (!complete()) {
        WARNING("Trying to construct incomplete operator.");
        return;
    }

    // Choose the most efficient (but equivalent) way to construct the matrix operator, given the size and filter configuration
    const unsigned N = _filter.n_elem;
    if ((N ==  2 && _size >= 2) ||
        (N <=  6 && _size >= 3) ||
        (N <= 20 && _size >= 4) ||
        (           _size >= 5)) {
        _constructByIndices();
    } else {
        _constructByRows();
    }

    return;
}

void MatrixOperator::_constructByRows () {
    
    const unsigned nRows = (unsigned) pow(2, _size);
    const unsigned nCols = (unsigned) pow(2, _size + 1);
    const unsigned N = _filter.n_elem;
    
    // Initialise the matrix operator to all zeros.
    this->zeros(nRows, nCols);
    
    
    // Initialise single row matrix with the appropriately positioned filter coefficients
    arma::Row<double> v (nCols, arma::fill::zeros);
    for (unsigned i = 0; i < N; i++) {
        v( (i - N/2 + 1) % nCols ) += _filter(N - i - 1);
    }
    
    // Use that all rows in the matrix operator are identical modulo even-numbered shifts to construct the remaining rows by shifted versions of the base row.
    this->row(0) = v;
    for (unsigned i = nRows; i --> 1; ) {

        // Shift current row two places to the right.
        rowshift(v, 2);

        // And set as i'th row in matrix operator.
        this->row(i) = v;        

    }

    
    return;
}

void MatrixOperator::_constructByIndices () {
    
    const unsigned nRows = (unsigned) pow(2, _size);
    const unsigned nCols = (unsigned) pow(2, _size + 1);
    const unsigned N = _filter.n_elem;
    
    // Initialise the matrix operator to all zeros.
    this->zeros(nRows, nCols);
    
    arma::Col<arma::uword> rows (nRows, arma::fill::zeros);
    arma::Col<arma::uword> cols (nRows, arma::fill::zeros);

    for (unsigned irow = 0; irow < nRows; irow++) {
        rows(irow) = irow;
    }
    for (unsigned icol = 0; icol < nRows; icol++) {
        cols(icol) = (N / 2 + 2 * icol) % nCols;
    }

    arma::Col<arma::uword> indices, idxZero;
    for (unsigned i = 0; i < N; i++) {
        indices = cols * nRows + rows; 
        (*this)(indices) += _filter(i);

        // Shift indices.
        idxZero = find(cols == 0);
        cols(idxZero) += nCols;
        cols -= 1;
    }
    
    return;
}

void rowshift (arma::Row<double>& row, const int& shift) {
    
    const unsigned length = row.n_elem;
    
    if (shift > 0 && shift < length) {
        
        std::rotate(row.begin(), row.begin() + shift, row.end());
        
    } else {
        
        if (length <  abs(shift)) {
            rowshift(row, sign(shift) * (abs(shift) % length));
        }
        
        if (shift  <  0) {
            rowshift(row, length + shift);
        }
        
    }
    
    return;
    
}

} // namespace
