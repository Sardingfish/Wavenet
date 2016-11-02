#include "Wavenet/MatrixOperator.h"


namespace Wavenet {
    
void MatrixOperator::construct () {
    
    if (!complete()) {
        WARNING("Trying to construct incomplete operator.");
        return;
    }
    
    unsigned nRows = (unsigned) pow(2, _size);
    unsigned nCols = (unsigned) pow(2, _size + 1);
    unsigned N = _filter.n_elem;
    
    this->zeros(nRows, nCols);
    
    
    // METHOD 1
    /*
    arma::Row<double> v (nCols, fill::zeros);
    arma::Row<double> remainder (2);
    for (unsigned i = 0; i < N; i++) {
        v( (i - N/2 + 1) % nCols ) += _filter(N - i - 1);
    }
    
    for (unsigned i = 0; i < nRows; i++) {
        
        this->row(i) = v;

        // Shift vector two spaces to the right.
        rowshift(v, +2);

    }
    */
    
    
    // METHOD 2
    arma::Row<double> v (nCols, arma::fill::zeros);
    for (unsigned i = 0; i < N; i++) {
        v( (i - N/2 + 1) % nCols ) += _filter(N - i - 1);
    }
    
    this->row(0) = v;
    for (unsigned i = nRows; i --> 1; ) {
        std::rotate(v.begin(), v.begin() + 2, v.end());
        this->row(i) = v;
        
    }

    
    // METHOD 3
    // Matrices are column by column
    /*
    this->zeros(nCols, nRows);
    uvec idx = linspace< uvec > ((- N/2) % nCols, nRows * nCols - 2 + (- N/2) % nCols, nRows);
    idx.print();
    for (unsigned i = 0; i < N; i++) {
        idx(nRows - 1) = idx(nRows - 1 ) % (nCols * nRows);
        this->elem(idx) = arma::Row<double> (nRows, fill::ones) * _filter(N - i - 1);
        idx = idx + 1;
    }
    inplace_trans(*this);
    */

    // METHOD 4
    /*
    uvec idxRow = linspace< uvec >(0, nRows - 1, nRows);
    uvec idxCol = linspace< uvec >(0, nCols - 2, nRows) + ( - N/2 ) % nCols;
    for (unsigned i = 0; i < N; i++) {
        idxCol = idxCol + 1;
        //idxCol = idxCol - idxCol / nCols;
        idxCol.transform( [nCols](int idx) { return idx % nCols; } );
        this->elem(idxRow + nRows*idxCol) = arma::Row<double> (nRows, fill::ones) * _filter(N - i - 1);
    }
    */
    
    return;
}

void rowshift (arma::Row<double>& row, const int& shift) {
    
    unsigned length = row.n_elem;
    
    if (shift > 0 && shift < length) {

        /*
        arma::Row<double> remainder (shift);
        remainder = row( span(length - shift, length - 1) );
        row( span(shift, length - 1) ) = row( span(0, length - 1 - shift) );
        row( span(0, shift - 1) ) = remainder;
         */
        
        std::rotate(row.begin(), row.end() - shift, row.end());

        
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
