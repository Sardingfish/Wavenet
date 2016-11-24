#ifndef WAVENET_WAVENET_H
#define WAVENET_WAVENET_H

/**
 * @file   Wavenet.h
 * @author Andreas Sogaard
 * @date   17 November 2016
 * @brief  Class for Wavenet objects.
 */

// STL include(s).
#include <iostream> /* std::cout, std::istream, std::ostream */
#include <cstdio> /* snprintf */
#include <vector> /* std::vector */
#include <string> /* std::string */
#include <cmath> /* log2, exp */
#include <cassert> /* assert */
#include <utility> /* std::move */
#include <algorithm> /* std::max */
#include <cstdlib> /* system */

// Armadillo include(s).
#include <armadillo>

// Wavenet include(s).
#include "Wavenet/Utilities.h"
#include "Wavenet/Logger.h"
#include "Wavenet/LowpassOperator.h"
#include "Wavenet/HighpassOperator.h"
#include "Wavenet/Snapshot.h"
#include "Wavenet/CostFunctions.h"

// Convenient typedef for the activations from the 1D forward transform.
typedef arma::field< arma::Col<double> >              Activations1D_t;
// Convenient typedef for the activations from the 2D forward transform.
typedef std::vector< std::vector< Activations1D_t > > Activations2D_t;


namespace wavenet {

/**
 * Class for Wavenet objects.
 *
 * Wavenet objects are a combination of (deep) artificial neural networks and 
 * wavelet functions/transforms. In particular, we use that (discrete) wavelet 
 * functions can be expressed in terms of a set of filter coefficients, which
 * satisfy five conditions. These conditions can be cast in the form of 
 * quadratic regularisation terms, and an objective function for any given set 
 * of filter coefficient can be formulated as some (differentiable) measure of
 * the sparsity with which it can represent a given class of input data. 
 * Furthermore, utilising the parallels between the matrix formulation of the 
 * wavelet transform and fully connected neural network, we can construe the 
 * wavelet transform as a complex, dyadic neural network, and use stochastic  
 * gradient descend with backpropagation to optimize the filter coefficients in 
 * terms of the joint regularisation and sparsity objective function. This joint
 * interpretation means that, instead of optimising each entry in the neural
 * network weight matrices (which are independent and numerous), the weight
 * matrices are highly restricted by identifying them with the low- and high-
 * pass filter matrices of the wavelet transform, such that the total number of
 * free parameters in the wavenet is equal to the number of filter coefficients 
 * in the wavelet transform.
 *
 * As familiar from the wavelet domain, the Wavenet class has as members a set
 * of filter coefficients to be optimised, as well as methods for constructing 
 * and caching low- and high-pass matrix operators. From the neural network 
 * domain the class has as members a regularisation constant (lambda), a 
 * learning rate (alpha) as well more advanced learning options such as batch 
 * gradient descend, momentum, adaptive learning rates and batch sizes, as well 
 * as a version of simulated annealing.
 *
 * Since the transform and backprogation methods are completely dynamic, no
 * fixed shape/architecture needs be enforced on any Wavenet object. The 
 * optimisation is simply performed on whatever input data is supplied (provided 
 * the shape matches the dyadic requirements of the wavelet transform).
 */
class Wavenet : Logger {

public:
     
/// Constructor(s).
    Wavenet () {};

    Wavenet (const double& lambda) :
        m_lambda(lambda)
    {};

    Wavenet (const double& lambda, const double& alpha) :
        m_lambda(lambda), m_alpha(alpha)
    {};

    Wavenet (const Wavenet& other) :
        m_lambda(other.m_lambda),
        m_alpha(other.m_alpha),
        m_inertia(other.m_inertia),
        m_inertiaTimeScale(other.m_inertiaTimeScale),
        m_filter(other.m_filter)
    {};
    

/// Destructor.
    ~Wavenet () {};
    

/// Get method(s).
    // Returns the regularisation constant (lambda).
    inline double lambda () const { return m_lambda; }
    // Returns the learning rate (alpha).
    inline double alpha () const { return m_alpha; }
    // Return the inertia.
    inline double inertia () const { return m_inertia; }
    // Returns the inertia time scale.
    inline double inertiaTimeScale () const { return m_inertiaTimeScale; }
    
    // Returns the vector of filter coefficients.
    inline arma::Col<double> filter () const { return m_filter; }
    // Returns the vector momentum in filter coefficient space.
    inline arma::Col<double> momentum () const { return m_momentum; }

    // Returns the batch size.
    inline int batchSize () const { return m_batchSize; }

    // Returns whether the wavenet is configured to learn wavelet functions. 
    inline bool wavelet () const { return m_wavelet; }

    // Returns the filter log. Returns a _reference_ to the filter log and 
    // doesn't have a const qualifier, such that the log can be modified 
    // externally.
    inline std::vector< arma::Col<double> >& filterLog () { return m_filterLog; }
    // Clears the filter log.
    inline void clearFilterLog () { return m_filterLog.clear(); }

    // Returns the cost log. Returns a _reference_ to the cost log and doesn't 
    // have a const qualifier, such that the log can be modified externally, 
    // e.g. by the Coach removing the last entry after training.
    inline std::vector< double >& costLog () { return m_costLog; }
    // Clear the cost log.
    inline void clearCostLog () { return m_costLog.resize(1, 0); }

    // Returns the first entry in the cost log.
    double firstCost () const;
    // Returns the last (non-zero) entry in the cost log.
    double lastCost () const;

    
/// Set method(s).
    // Set the regularisation constant (lambda).
    inline bool setLambda (const double& lambda) {
       assert(lambda >= 0);
       m_lambda = lambda;
       return true;
    }
    // Set the learning rate (alpha).
    inline bool setAlpha (const double& alpha) {
        assert(alpha > 0);
        m_alpha = alpha;
        return true;
    }
    // Set the inertia.
    inline bool setInertia (const double& inertia) {
        assert(inertia >= 0 && inertia < 1);
        m_inertia = inertia;
        return true;
    }
    // Set the inertia time scale.
    inline bool setInertiaTimeScale (const double& inertiaTimeScale) {
        assert(inertiaTimeScale > 0);
        m_inertiaTimeScale = inertiaTimeScale;
        return true;
    }

    // Set the vector of filter coefficients.
    bool setFilter (const arma::Col<double>& filter);
    // Set the vector momentum in filter coefficient space.
    inline bool setMomentum (const arma::Col<double>& momentum) {
        assert(momentum.size() == m_filter.size());
        m_momentum = momentum;
        return true;
    }

    // Set the batch size.
    inline bool setBatchSize (const unsigned& batchSize) {
        m_batchSize = batchSize;
        return true;
    }

    // Specify whether the wavenet should learn wavelet functions
    inline bool doWavelet (const bool& wavelet) {
        m_wavelet = wavelet;
        return true;
    }
    

/// Print method(s).
    /**
     * @brief Print the configuration of the wavenet object to stdout.
     */
    void print () const;


/// Storage method(s).
    /**
     * @brief Save the wavenet object to file.
     *
     * Save the wavenet object to the file pointed to by the Snapshot object.
     *
     * Snapshots are passed by value since (1) they're tiny, (2) the methods are
     * called infrequently, and (3) this allows us load from/save to temporary 
     * snapshot objects, which means that we can do e.g. wavenet.save(snap++).
     * 
     * @param snap The Snapshot object to which the wavenet object is to be 
     *             saved.
     */
    void save (Snapshot snap) const;

    /**
     * @brief Load a wavenet object from file.
     *
     * Load a wavenet object from the file pointed to by the Snapshot object. 
     * This method modifies the internal state of the calling Wavenet object to 
     * match that of the saved object.
     *
     * @param snap The Snapshot object from  which the wavenet object is to be 
     *             loaded.
     */
    void load (Snapshot snap);


/// High-level learning method(s).
    /**
     * @brief Main method for training wavenet instance. 
     * 
     * This method (1) forward propagates the input example to find the 
     * activations of all nodes, (2) computes the sparsity error gradient on the
     * resulting wavelet coefficients, (3) computes the regularisation error 
     * gradient on the current member filter coeffients, (4) back-propagates the 
     * combined (sparsity and regularisation) gradient through the wavenet and 
     * accumulates the error gradient associated with each filter coefficient 
     * using weight matrices, and finally (5) appends the combined vector 
     * gradient in filter coefficient space to the batch queue. If the batch 
     * queue has reached the target size (which may be 1, with default setting, 
     * in which case batch gradient descend is not used), the method will also 
     * trigger an update of the wavenet object by flushing the batch queue.
     * 
     * @see forward_(...)
     * @see backpropagate_(...)
     * @see flushBatchQueue_()
     * 
     * @param X Input data example, on which to train the wavenet object.
     */
    bool train (arma::Mat<double> X);  

    /**
     * @brief Clear all non-essential data from wavenet object.
     * 
     * This methods scales the filter coefficient momentum to zero and clears 
     * the filter log, the cost log, and all cached matrix operators.
     */
    void clear ();

    
/// High-level cost method(s).
    /**
     * @brief Compute the combined cost of a vector of wavelet coefficients, and
     *        the member filter coefficients.
     *
     * The method comptued the combined cost of the current wavenet 
     * configuration with wavelet coefficients specified as argument. The 
     * sparsity term is computed as the Gini coefficient of the wavelet 
     * coefficiets. The regularisation term is computed as squared deviations 
     * from the five (four non-trivial) wavelet conditions on the filter 
     * coefficients. The regularisation term is scaled by the regularisation 
     * constant lambda.
     *
     * @param y A vector of wavelet coefficients, from the forward transform.
     * @return The combined (sparsity + regularisation) cost of the wavenet 
     *         configuration given the specified wavelet coefficients.
     */
    double cost (const arma::Col<double>& y);

    /**
     * @brief Compute the combined cost of a matrix of wavelet coefficients, and
     *        and the member filter coefficients.
     *
     * Wraps the cost method accepting vectors of wavelet coefficients.
     * @see cost(arma::Col<double>)
     * @param Y A matrix of wavelet coefficients, from the forward transform.
     * @return The combined (sparsity + regularisation) cost of the wavenet 
     *         configuration given the specified wavelet coefficients.   
     */
    double cost (const arma::Mat<double>& Y);
    
    /**
     * @brief Produce cost maps for a set of input examples.
     *
     * This method performs a 2D scan of filter coefficients, and for each 
     * combination computes the combined, sparsity-only, and regularisation-only 
     * costs given the provided input examples. This is useful for visualising 
     * the low-dimensionality cost space of the data, but due to visualisation 
     * and computating limitations, the method can only scan filter coefficients 
     * in two dimensions. (This is core of the argument for needing a machine 
     * learning techinque for higher-dimension optimisation in the first place).
     *
     * @param X A vector of input examples, in the form of Armadillo matrices, 
     *          for which the cost map is to be produced.
     * @param range Upper and lower (-range) limits of the filter coefficients 
     *              used to compute the cost map. A range of approx. 1 is likely
     *              sufficient, due to the conditions of orthonormality imposed
     *              on the filter coefficients.
     * @param Ndiv The number of divisions along each axis in scanning the 
     *             filter coefficients. The output matrix will have dimensions
     *             Ndiv x Ndiv. A larger number of divisions means greater 
     *             granularity, but the run time grows as Ndiv squared.
     * @return Three matrices, mapping the (1) combined cost, (2) sparsity cost,
     *         and (3) regularisation cost for the scan of filter coefficients
     *         given the set of input examples. These maps are stored as 
     *         Armadillo matrices, where each entry corresponds to a cost for a
     *         given filter coefficient configuration, which can be found 
     *         explicitly from the specified range and number of divisions. Each
     *         map is stored in an Armadillo field, and is accessed as map(0,0),
     *         map(1,0), and map(2,0), resp.          
     */
    arma::field< arma::Mat<double> > costMap (const std::vector< arma::Mat<double> >& X,
                                              const double& range,
                                              const unsigned& Ndiv);
    

/// Basis function method(s).
    /**
     * @brief Generate 1D basis funtion.
     * 
     * Given a specified (1D) wavelet coefficient, generate the corresponding 
     * (1D)position space basis function by setting this coefficient to one, 
     * the remaining coefficients to zero, and performing the inverse transform.
     * 
     * @param nRows The number of wavelet coefficients from which to choose.
     * @param irow The wavelet coefficient index for the basis function.
     * @return An Armadillo matrix of size nRows x 1 containing the position 
     *         space basis function corresponding to the specified wavelet  
     *         coefficient index.
     */
    arma::Mat<double> basisFunction1D (const unsigned& nRows, const unsigned& irow);

    /**
     * @brief Generate 2D basis funtion.
     * 
     * Given a specified (2D) wavelet coefficient, generate the corresponding
     * (2D) position space basis function by setting this coefficient to one, 
     * the remaining coefficients to zero, and performing the inverse transform.
     * 

     * @param nRows The number of rows in the set of wavelet coefficients from 
     *              which to choose.
     * @param nCols The number of columns in the set of wavelet coefficients 
     *              from which to choose.
     * @param irow The wavelet coefficient index along the row axis for the 
     *             basis function.
     * @param icol The wavelet coefficient index along the column axis for the 
     *             basis function.
     * @return An Armadillo matrix of size nRows x nCols containing the position 
     *         space basis function corresponding to the specified wavelet 
     *         coefficient indices.
     */
    arma::Mat<double> basisFunction2D (const unsigned& nRows, const unsigned& nCols,
                                       const unsigned& irow,  const unsigned& icol);

    /**
     * @brief Generate basis function.
     * 
     * Wrapper around the specialised basis function methods. Depending on the 
     * size of the wavelet coefficient matrix (nRows x nCols; in particular, 
     * whether nRows > 1 and nCols > 1), calls one or the other and returns the 
     * resulting position space basis function matrix.
     *
     * @see basisFunction1D(...)
     * @see basisFunction2D(...)
     */
    arma::Mat<double> basisFunction (const unsigned& nRows, const unsigned& nCols,
                                     const unsigned& irow,  const unsigned& icol);
    

protected: 

/// 1D wavenet transform method(s).
    /**
     * @brief Forward transform of (column) vector.
     *
     * Since C++ methods cannot be distinguished simply based on the type of the
     * return argument, no equivalent method returning a vector of wavelet 
     * coefficients is implemented. This behaviour can however be achieved using 
     * the 'coeffsFromActivations' method (@see Utilities.h).
     *
     * @param x The input vector which is forward transformed.
     * @return Collection of neural network activations for all layers/levels 
     *         in the wavenet.
     */
    Activations1D_t forward_ (const arma::Col<double>& x);
    
   /**
     * @brief Inverse transform of vector of wavelet coefficients.
     *
     * Perform the inverse wavelet transform of a set of wavelet coefficients.
     * This is _not_ the same as performing the neural network backpropagation. 
     * A similar inverse transform of a collection of neural network activations
     * back be perform by using the 'coeffsFromActivations' utility function.
     *
     * @see backpropagate_(arma::Col<double>, Activations1D_t)
     * @see forward_(arma::Col<double>)
     *
     * @param activations Vector of wavelet coefficients.
     * @return Vector of values in position space.
     */
    arma::Col<double> inverse_ (const arma::Col<double>& y);

    /**
     * @brief Backpropagate 1D errors through wavenet.
     *
     * Given a set of computed errors, according to some optimality criterion 
     * (here: sparsity), of a set of wavelet coefficients, and a collection of 
     * the neural network activations for the 1D forward transform into this set 
     * of wavelet coefficients, propagate the errors back through the wavenet. 
     * In this way, each entry in the matrix operators, and by implication the
     * wavelet filter coefficients, are assigned (several) error terms, the sum 
     * of which indicates the direction of steepest descend in the space of 
     * filter coefficients.
     *
     * @param delta The computed errors on the wavelet coefficient vector for a 
     *              given input.
     * @param activations The 1D neural network activations, used for 
     *                    progagating the errors back through the wavenet.
     * @return An Armadillo field with two entries: (1) The errors on the input,
     *         used when perforing the backpropagation of a 2D wavenet, where
     *         the errors on the "input" for the rows are used as the base 
     *         errors for the backpropagation for each column. (2) A vector of 
     *         the error gradients for the filter coefficents.
     */
    std::vector< arma::Col<double> > backpropagate_ (const arma::Col<double>& delta,
                                                     Activations1D_t activations);


/// 2D wavenet transform method(s).
    /**
     * @brief Forward transform of matrix.
     *
     * The 2D forward wavelet transform is implemented in a row-major fashion: 
     * Each row is individually forward transformed using the corresponding 1D 
     * method; the columns of the resulting matrix are then similarly forward 
     * transformed.
     *
     * @see forward_(arma::Col<double>)
     *
     * @param X The input matrix which is forward transformed.
     * @return Collection of neural network activations for all layers/levels 
     *         in the wavenet.
     */
    Activations2D_t   forward_ (const arma::Mat<double>& X);
    
    /**
     * @brief Inverse transform of matrix of wavelet coefficient.
     *
     * As the  2D forward wavelet transform is implemented in a row-major 
     * fashion, the inverse transform is implemented in column-major fashion: 
     * Each column is individually inverse transformed using the corresponding 
     * 1D method; the rows of the resulting matrix are then similarly inverse 
     * transformed.
     *
     * @see inverse_(arma::Col<double>)
     *
     * @param X Matrix of wavelet coefficients.
     * @return Matrix of values in position space.
     */
    arma::Mat<double> inverse_ (const arma::Mat<double>& Y);
    
    /**
     * @brief Backpropagate 2D errors through wavenet.
     *
     * Propagate the computed errors on the wavelet coefficient backwards 
     * through the wavenet. This is done by iteratively applying the 1D version
     * of the method in the opposite order of the forward transform. Since the 
     * 2D forward transform is performed in a row-major fashion, the 
     * backpropagation is performed in a column-major fashion: Each column in 
     * the matrix of errors is individually backpropagated using the 1D 
     * backprogation method; the resulting rows in the matrix of errors are then
     * similarly backpropagated individually. For each 1D backpropagation 
     * operation, the gradients for the filter coefficients are summed.
     *
     * Contrary to the 1D version, we opt to return only the final, 
     * backpropagated gradients on the filter coefficients, not _both_ these
     * _and_ the "errors on the input", as these serve no purpose here.
     *
     * @see backpropagate_(arma::Col<double>, Activations1D_t)
     *
     * @param Delta The computed errors on the wavelet coefficient matrix for a 
     *              given input.
     * @param Activations The 2D neural network activations, used for 
     *                    progagating the errors back through the wavenet.
     * @return A vector of the error gradients for the filter coefficents.
     */
    arma::Col<double> backpropagate_ (const arma::Mat<double>& Delta,
                                      Activations2D_t Activations);


/// Low-level learning method(s).
    /**
     * @brief Flush the batch queue.
     * 
     * Averages the filter coefficient gradients in the batch queue, calls the 
     * update method with the average gradient, clears the batch queue, and 
     * appends the average cost of the examples in the batch to the cost log.
     * 
     * @see update_(arma::Col<double>)
     */
    void flushBatchQueue_ ();


    /**
     * @brief Add gradient to filter coefficient mometum.
     *
     * If a momentum already exists, add the gradient to it. Otherwise, create a 
     * momentum given by the inpu gradient.
     */
    void addMomentum_ (const arma::Col<double>& gradient);
   
    /**
     * @brief Scale the momentum
     * 
     * Multiply the (vector) filter coefficient momentum by a scalar factor.
     */
    void scaleMomentum_ (const double& factor);


    /**
     * @brief Update the filter coefficients with gradient.
     * 
     * The main, low-level method responsible for the learning of optimal filter
     * coefficients. The method scales existing momentum of the wavenet instance
     * by the inertia, adds the input gradient to the momentum, and updates the 
     * filter coefficients by the updated momentum vector. (Note: If the inertia
     * is zero, the momentum does nothing, and the filter coefficients are 
     * simply updated by the gradient.) If an inertia and an inertia time scale
     * are specified, an effictive inertia is computed before performing the 
     * momentum update.
     *
     * @see scaleMomentum_(double)
     * @see addMomentum_(arma::Col<double>) 
     * @see setFilter(arma::Col<double>) 
     *
     * @param gradient The vector gradient in filter coefficient space according 
     *                 to which to update the filter coefficients.
     */
    void update_ (const arma::Col<double>& gradient);
    

    /**
     * @brief Cache matrix operators.
     * 
     * Clears the existing cache of matrix operators, and stores vectors of low- 
     * and high-pass matrix operators from scale 0 to the specified scale m. 
     * This saves the work initialising identical LowpassOperator and 
     * HighpassOperator objects for each row and column in the 2D transforms.
     * The the smallest cached matrix operators will be of size 1 x 2, and 
     * largest will be of size 2^m x 2{m + 1}. 
     *
     * After successfully caching all requested matrix operators, the methods 
     * switches a flag to notify that the caching has taken place.
     *
     * @see clearCachedOperators_()
     *
     * @param m The scale (log2) up to which matrix operators should be cached. 
     */
    void cacheOperators_ (const unsigned& m);
   
    /**
     * @brief Clear matrix operator cache.
     * 
     * Resets cache vectors, and switch the flag to notify that the cache is 
     * empty.
     */
    void clearCachedOperators_ ();
    

    /**
     * @brief Cache matix weights.
     * 
     * Clears the existing cache of matrix weights, and stores vector computed 
     * low- and high-pass matrix weights from scale 0 to the specified scale m. 
     * The matrix weights are found, at each frequency scale between 0 and m 
     * inclusive, by setting all filter coefficients to zero, one by one 
     * switching each coefficient to value one, and constructing the resulting 
     * low- and high-pass matrix operators. These weights (all either +1 or -1) 
     * are then used to compute the weighted sum of the backprogated errors, for
     * each layer in the wavenet, which are attributed to each filter coefficient
     *
     * @see clearCachedWeights_()
     *
     * @param m The scale (log2) up to which matrix weights should be cached.
     */
    void cacheWeights_ (const unsigned& m);
   
    /**
     * @brief Clear matrix weights cache.
     * 
     * Resets cache vectors, and switch the flag to notify that the cache is 
     * empty.
     */
    void clearCachedWeights_ ();


    /**
     * @brief Apply low-pass filter.
     * 
     * Multiply the input vector with the cached low-pass matrix operator 
     * appropriate for the given layer.
     *
     * @param x The position space-like vector to be low-pass filtered.
     * @return The low-pass filtered vector.
     */
    arma::Col<double> lowpassfilter_ (const arma::Col<double>& x);
   
    /**
     * @brief Apply high-pass filter.
     * 
     * Multiply the input vector with the cached high-pass matrix operator 
     * appropriate for the given layer.
     *
     * @param x The position space-like vector to be high-pass filtered.
     * @return The high-pass filtered vector.
     */
    arma::Col<double> highpassfilter_ (const arma::Col<double>& x);
    
    /**
     * @brief Apply inverse low-pass filter.
     * 
     * Multiply the input vector with the transpose of the cached low-pass 
     * matrix operator appropriate for the given layer.
     *
     * @param y The momentum space-like vector to be inverse low-pass filtered.
     * @return The inverse low-pass filtered vector.
     */
    arma::Col<double> inv_lowpassfilter_ (const arma::Col<double>& y);
    
    /**
     * @brief Apply inverse high-pass filter.
     * 
     * Multiply the input vector with the transpose of the cached high-pass 
     * matrix operator appropriate for the given layer.
     *
     * @param y The momentum space-like vector to be inverse high-pass filtered.
     * @return The inverse high-pass filtered vector.
     */
    arma::Col<double> inv_highpassfilter_ (const arma::Col<double>& y);

    
    /**
     * @brief Get the low-pass weight matrix.
     * 
     * Get the cached low-pass weight matrix for the filter coefficient with 
     * index 'filt' at frequency scale 'level'.
     *
     * @param level The frequency scale for which to get the weight matrix.
     * @param filt The index of the filter coefficient for which to get the 
     *             weight matrix.
     * @return The cached low-pass weight matrix.
     */
    const arma::Mat<double>& lowpassweight_ (const unsigned& level,
                                             const unsigned& filt);
    
    /**
     * @brief Get the high-pass weight matrix.
     * 
     * Get the cached high-pass weight matrix for the filter coefficient with 
     * index 'filt' at frequency scale 'level'.
     *
     * @param level The frequency scale for which to get the weight matrix.
     * @param filt The index of the filter coefficient for which to get the 
     *             weight matrix.
     * @return The cached high-pass weight matrix.
     */
    const arma::Mat<double>& highpassweight_ (const unsigned& level,
                                              const unsigned& filt);


protected:

/// Streaming operator(s).
    /**
     * As these methods relate chiefly to the Snapshot class, they are 
     * implemented in Snapshot.cxx.
     */
    friend       Snapshot& operator<< (      Snapshot& snap, const Wavenet& wavenet);
    friend const Snapshot& operator>> (const Snapshot& snap,       Wavenet& wavenet);


private: 
    
/// Data member(s).
    // Learning parameter member(s).
    /**
     * @brief The regularisation constant, lambda.
     * 
     * Lambda controls the relative contribution of the regularisation term to 
     * the combined cost, the other being the sparsity term. Larger values of
     * lambda means that wavelet (pseudo-)bases found from training will be 
     * closer to being actual, exact orthonormal bases. However, larger values
     * of lambda also means steeper gradients and deeper, narrower "minima 
     * valleys", which will more easily lead to slow or diverging solutions.
     *
     * The 'useSimulatedAnnealing' methods allows for a regularisation constant
     * which starts out at zero and grows with time. This might help to avoid 
     * early diverging solution, and avoid local regularisation-dominated 
     * minima.
     */
    double m_lambda = 10.0;

    /**
     * @brief The learning rate, alpha.
     * 
     * The factor multiplying (regulating) the gradient, when updating the 
     * filter coefficients. Larger values of alpha means faster traversal of the 
     * filter coefficient space and therefore faster solutions. However, larger
     * value of alpha might also lead to diverging solution (when combined with
     * sufficitently large values of lambda) and will lead to less precise 
     * solutions. Good, stable values seem to be > 0.001. 
     *
     * The 'useAdeptiveLearningRate' method allows for alpha to decrease as 
     * minima are reached. This would allow for fast traversal of the filter 
     * coefficient space early in the optimisation, as well as precise solution 
     * towards the end.
     */
    double m_alpha = 0.001;
    
    /**
     * @brief The filter coefficient space momentum inertia.
     * 
     * The factor by which the filter coefficient space is multiplied, before 
     * adding a new, batch-averaged gradient. Controls how the contribution from
     * earlier steps in the optimisation process decays with time. A value of 0
     * means that the learning has no memory of previous updates, and each 
     * update of the filter coefficients is solely based on the current 
     * gradient. A value of 1 (not allowed) means that all previous gradients 
     * would continue to contribute exactly the same amout at all future times.
     * This would prevent the optimisation to find stable minima. Some 
     * intermediate value can make the traversal of filter coefficient space 
     * faster and more stable in cases of steep cost contours (large values of 
     * lambda) and may help to avoid local, sub-optimal minima.
     */
    double m_inertia = 0.;
    
    /**
     * @brief The time scale of inertia onset.
     * 
     * Given the steep cost contours associated with the regularisation terms, 
     * starting out with a large (> 0.9) inertia at a randomly generated point 
     * in filter coefficient space may lead to divergences. Setting an inertia 
     * time scale means that the inertia will start out at zero, allowing the 
     * Wavenet to perform the first few, steep updates with no inertia, but will 
     * then grow to the specified inertia value as 
     *   i_{eff} = i * (1 - \exp(-n/\tau))
     * where i is the specified inertia value, \tau is the inertia time scale,
     * n is the number of steps completed.
     */
    double m_inertiaTimeScale = 0.;
    

    // Filter coefficient space member(s).
    /**
     * @brief The vector of filter coefficients.
     */
    arma::Col<double> m_filter;
    
    /**
     * @brief The filter coefficient space momentum.
     */
    arma::Col<double> m_momentum;


    // Cached matrix operator member(s).
    /**
     * @brief Whether the instance has cached matrix operators.
     */
    bool m_hasCachedOperators = false;
    
    /**
     * @brief Container of cached low-pass matrix operators.
     */
    arma::field< arma::Mat<double> > m_cachedLowpassOperators;
    
    /**
     * @brief Container of cached high-pass matrix operators.
     */
    arma::field< arma::Mat<double> > m_cachedHighpassOperators;


    // Cached matrix weights member(s).
    /**
     * @brief Whether the instance has cached matrix weights.
     */
    bool m_hasCachedWeights = false;
    
    /**
     * @brief Container of cached low-pass matrix weights.
     */
    arma::field< arma::Mat<double> > m_cachedLowpassWeights;
    
    /**
     * @brief Container of cached high-pass matrix weights.
     */
    arma::field< arma::Mat<double> > m_cachedHighpassWeights;
    

    // Learning container member(s).
    /**
     * @brief The size of the batches used in the gradient descend.
     * 
     * If the batch size is kept at 1, no batch gradient descend is performed 
     * and each training example will trigger an update of the filter 
     * coefficients. If the batch size is larger than one, the filter 
     * coefficient gradient arising from each training example will be stored in 
     * a batch queue, until the batch queue reaches the specified size, and the 
     * Wavenet object is updated by the batch-averaged gradient. This vill lead
     * to a more stable, but slower, learning proces.
     */
    unsigned m_batchSize = 1;
    
    /**
     * @brief The batch queue.
     * 
     * Contains one filter coefficient space gradient for each example in the 
     * current batch, to be added and scaled when doing a batch update.
     */
    std::vector< arma::Col<double> > m_batchQueue;
    
    /**
     * @brief The filter log.
     * 
     * The log or history of filter coefficients throughout the training, with 
     * one entry for each update/learning step.
     */
    std::vector< arma::Col<double> > m_filterLog;
    
    /**
     * @brief The cost log.
     * 
     * The log or history of combined costs (sparsity and regularisation) 
     * throughout the training with one entry for each update/learning step.
     */
    std::vector< double > m_costLog = {0};
    

    // Function type members(s).
    /**
     * @brief Whether the instane is configured to learn wavelet functions.
     * 
     * Alternative is any orthonormal function basis which can be expressed in
     * terms of identical, successive filter (but not necessarily low- and high-
     * pass type filters.)
     */
    bool m_wavelet = true;
    
};

/// Utility function(s).
/**
 * @brief Frobenius inner product between matrices A and B. 
 * 
 * Used for computing the error on the weight matrix entries, i.e. the wavelet 
 * coefficients. In the current design, this is the most significant performance 
 * bottleneck.
 */ 
template<class T>
inline T frobeniusProduct (const arma::Mat<T>& A, const arma::Mat<T>& B) { 
    // Other, slower variants:
    //   return arma::trace( A * B.t() ); 
    //   return arma::as_scalar( arma::vectorise(A).t() * arma::vectorise(B) ); 
    return arma::accu( A % B );
};

/**
 * @brief Vector outer product between vectors a and b.
 * 
 * Construct a matrix from the outer product of two vectors.
 */
template<class T>
inline arma::Mat<T> outerProduct (const arma::Col<T>& a, const arma::Col<T>& b) { 
    return (a * b.t()); 
}

} // namespace

#endif // WAVENET_WAVENET_H
