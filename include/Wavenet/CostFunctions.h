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

/// Sparsity function(s).
/**
 * @brief Compute the sparsity term (Gini coefficient).
 * 
 * ...
 *
 * @param y A vector of wavelet coefficients.
 * @returns The Gini coefficient of the input
 */
double SparseTerm (const arma::Col<double>& y);
/**
 * @brief Compute the sparsity term (Gini coefficient).
 * 
 * ...
 *
 * @param Y A matrix of wavelet coefficients.
 * @returns The Gini coefficient of the input
 */
 double SparseTerm (const arma::Mat<double>& Y);

// Computing the derivative of the sparsity term (Gini coefficient).
arma::Col<double> SparseTermDeriv (const arma::Col<double>& y);
arma::Mat<double> SparseTermDeriv (const arma::Mat<double>& Y);

/// Regularisation function(s).
// Computing the regularisation term.
double RegTerm (const arma::Col<double>& y, const bool& doWavelet = true);

// Computing the derivative of the regularisation term.
arma::Col<double> RegTermDeriv (const arma::Col<double>& y, const bool& doWavelet = true);

} // namespace

#endif // WAVENET_COSTFUNCTIONS_H
