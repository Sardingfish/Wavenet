#ifndef WAVENET_COSTFUNCTIONS_H
#define WAVENET_COSTFUNCTIONS_H

/**
 * @file   CostFunctions.h
 * @author Andreas Sogaard
 * @date   15 November 2016
 * @brief  ...
 */

// STL include(s).
// ...

// Armadillo include(s).
#include <armadillo>

// Wavenet include(s).
#include "Wavenet/MatrixOperator.h"
 

namespace wavenet {

/// Computing the sparsity term (Gini coefficient).
double SparseTerm (const arma::Col<double>& y);
double SparseTerm (const arma::Mat<double>& Y);

/// Computing the derivative of the sparsity term (Gini coefficient).
arma::Col<double> SparseTermDeriv (const arma::Col<double>& y);
arma::Mat<double> SparseTermDeriv (const arma::Mat<double>& Y);

/// Computing the regularisation term.
double RegTerm (const arma::Col<double>& y, const bool& doWavelet = true);

/// Computing the derivative of the regularisation term.
arma::Col<double> RegTermDeriv (const arma::Col<double>& y, const bool& doWavelet = true);

} // namespace

#endif // WAVENET_COSTFUNCTIONS_H
