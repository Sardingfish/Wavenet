#ifndef WAVENET_COSTFUNCTIONS_H
#define WAVENET_COSTFUNCTIONS_H

/**
 * @file   CostFunctions.h
 * @author Andreas Sogaard
 * @date   15 November 2016
 * @brief  Collection of cost functions.
 */

// Armadillo include(s).
#include <armadillo>

// Wavenet include(s).
#include "Wavenet/MatrixOperator.h"


namespace wavenet {

/// Sparsity function(s).
/**
 * @brief Compute the sparsity term (Gini coefficient).
 * 
 * The objective function of the wavenet is chosen such that it minimises 
 * _sparsity_, i.e. how well a certain functional basis is able to represent a 
 * given class of input. In particular, the Gini coefficient is chosen as the 
 * metric of sparsity, and the sparsity term in the objective function is chosen 
 * to be:
 *   S({c}) = 1 - G({c})
 * where G({c}) is the Gini coefficient of the set of wavelet coefficients {c} 
 * for a given position-space training example. This is the value return by the 
 * method.
 *
 * @param c Vector of wavelet coefficients.
 * @return The Gini coefficient of the input
 */
double SparseTerm (const arma::Col<double>& c);

/**
 * @brief Compute the sparsity term (Gini coefficient).
 * 
 * Wrapper around SparseTerm(arma::Col<double>).
 *
 * @param C Matrix of wavelet coefficients.
 * @return The Gini coefficient of the input
 */
 double SparseTerm (const arma::Mat<double>& C);


/**
 * @brief Compute gradient on wavelet coefficients from sparsity.
 * 
 * This method computes the gradient of the sparsity term with respect to each
 * of the wavelet coefficients in the expression for the Gini coefficient. This
 * gives the errors on each of the input wavelet coefficients, which are then
 * propagated backwards through the wavenet in order to translate these errors
 * into gradients on the filter coefficients so as to minimise the sparisty
 * objective function.
 *
 * @param c Vector of wavelet coefficients.
 * @return Vector gradient on the input wavelet coefficients according to the
 *         sparsity objective function.
 */
arma::Col<double> SparseTermDeriv (const arma::Col<double>& c);

/**
 * @brief Compute gradient on wavelet coefficients from sparsity.
 * 
 * Wrapper around SparseTermDeriv(arma::Col<double>).
 *
 * @param C Matrix of wavelet coefficients.
 * @return Matrix gradient on the input wavelet coefficients according to the
 *         sparsity objective function.
 */
arma::Mat<double> SparseTermDeriv (const arma::Mat<double>& C);


/// Regularisation function(s).
/*
 * @brief Compute the regularisation term.
 *
 * The filter coefficients in the wavenet are subjected to quadratic 
 * regularisation constraints to ensure that they satisfy conditions (C1-5) in 
 * the companion note. These regularisation constrains ensure that the filter 
 * coefficients do indeed produce a wavelet (pseudo-)basis. This methods 
 * computes the sum of the five (four actual) regularisation terms given a 
 * vector of filter coefficients.
 *
 * The method also allows for omitting wavelet-specific regularisation 
 * constraints (C1, C3, and C4). If this is done (doWavelet = false) a general 
 * functional basis found from training may be able to represent the class of 
 * training data better, with greater sparsity, but will not constitute a 
 * wavelet basis. However, in these use cases, a generic neural network might be 
 * a better choice.
 *
 * @param a Vector of filter coefficients.
 * @param doWavelet Whether to impose wavelet-specific regulsarisation.
 * @return The combined regularisation term.
 */
double RegTerm (const arma::Col<double>& a, const bool& doWavelet = true);

/**
 * @brief Compute gradient on filter coefficients from regularisation.
 *
 * This method computes the gradient of the five (four) regularisation terms 
 * with respect to each of the input filter coefficients. The resulting gradient 
 * is used directly in the stochastic gradient descend in filter space, along 
 * with the sparsity gradient found by backpropagating the errors on wavelet 
 * coefficients. This method returns the bare gradient, which can (out to) be 
 * regulated by a regularisation term, lambda.
 *
 * The @c doWavelet flag indicates whether to include gradients from the 
 * wavelet-specific regularisation terms (C1, C3, and C4).
 * 
 * @see SparseTermDeriv(arma::Col<double>)
 *
 * @param a Vector of filter coefficients.
 * @param doWavelet Whether to impose wavelet-specific regularisation.
 * @return Vector gradient in filter coefficient space.
 */
arma::Col<double> RegTermDeriv (const arma::Col<double>& a, const bool& doWavelet = true);

} // namespace

#endif // WAVENET_COSTFUNCTIONS_H
