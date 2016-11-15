#ifndef WAVENET_WAVENET_H
#define WAVENET_WAVENET_H

/**
 * @file Wavenet.h
 * @author Andreas Sogaard
 */

// STL include(s).
#include <iostream> /* std::cout, std::istream, std::ostream */
#include <cstdio> /* snprintf */
#include <vector> /* std::vector */
#include <string> /* std::string */
#include <cmath> /* log2 */
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

namespace wavenet {

class Wavenet : Logger {

public:
    
    /**
     * @TODO: Declare all member function which do not change the state of the of the object (i.e. all get-methods) as 'const'.
     */
    
    /// Constructor(s).
    Wavenet () {};
    Wavenet (const double& lambda) :
        m_lambda(lambda)
    {};
    Wavenet (const double& lambda, const double& alpha) :
        m_lambda(lambda), m_alpha(alpha)
    {};
    Wavenet (const double& lambda, const double& alpha, const double& inertia) :
        m_lambda(lambda), m_alpha(alpha), m_inertia(inertia)
    {};
    Wavenet (const double& lambda, const double& alpha, const double& inertia, const double& inertiaTimeScale) :
        m_lambda(lambda), m_alpha(alpha), m_inertia(inertia), m_inertiaTimeScale(inertiaTimeScale)
    {};
    Wavenet (const Wavenet& other) :
        m_lambda(other.m_lambda), m_alpha(other.m_alpha), m_inertia(other.m_inertia), m_inertiaTimeScale(other.m_inertiaTimeScale), m_filter(other.m_filter)
    {};
    

    /// Destructor.
    ~Wavenet () {};
    

    /// Get method(s).
    inline double lambda           () const { return m_lambda; }
    inline double alpha            () const { return m_alpha; }
    inline double inertia          () const { return m_inertia; }
    inline double inertiaTimeScale () const { return m_inertiaTimeScale; }
    
    inline arma::Col<double> filter   () const { return m_filter; }
    inline arma::Col<double> momentum () const { return m_momentum; }

    inline int batchSize () const { return m_batchSize; }

    inline bool wavelet () const { return m_wavelet; }

    inline std::vector< arma::Col<double> >      filterLog () const { return m_filterLog; }
    inline void                             clearFilterLog ()       { return m_filterLog.clear(); }

    inline std::vector< double >      costLog () const { return m_costLog; }
    inline void                  clearCostLog ()       { return m_costLog.resize(1, 0); }

    inline double lastCost () const { return (costLog().size() > 0 ? costLog()[costLog().size() - 1]: -1);}

    
    /// Set method(s).
    inline bool setLambda           (const double& lambda)           { assert(lambda           > 0); m_lambda           = lambda;           return true; }
    inline bool setAlpha            (const double& alpha)            { assert(alpha            > 0); m_alpha            = alpha;            return true; }
    inline bool setInertia          (const double& inertia)          { assert(inertia          > 0); m_inertia          = inertia;          return true; }
    inline bool setInertiaTimeScale (const double& inertiaTimeScale) { assert(inertiaTimeScale > 0); m_inertiaTimeScale = inertiaTimeScale; return true; }

           bool setFilter   (const arma::Col<double>& filter);
    inline bool setMomentum (const arma::Col<double>& momentum) { assert(momentum.size() == m_filter.size()); m_momentum = momentum; return true; }

    inline bool setBatchSize (const unsigned& batchSize) { m_batchSize = batchSize; return true; }

    inline bool doWavelet (const bool& wavelet) { m_wavelet   = wavelet;   return true; }
    

    /// Print method(s).
    void print () const;


    /// Storage method(s).
    /**
     * The snapshots are passed by value since (1) they're tiny, (2) the methods are called infrequently, and (3) this allows us load from/save to temporary snapshot objects, which means that we can do e.g. wavenet.save(snap++).
     */
    void save (Snapshot snap) const;
    void load (Snapshot snap);

    
    /// 1D wavenet transform method(s).
    /**
     * @brief Forward transform of (column) vector.
     *
     * Since C++ methods cannot be distinguished simply based on the type of the
     * return argument, no equivalent method returning a vector of wavelet 
     * coefficients is implemented. This behaviour can however be achieved using 
     * the 'coeffsFromActivations' method below.
     *
     * @param x The input vector which is forward transformed.
     * @returns Collection of neural network activations for all layers/levels 
     *          in the wavenet.
     */
    arma::field< arma::Col<double> > forward (const arma::Col<double>& x);
    
    /**
     * @brief Inverse transform of collection of neural network activations.
     *
     * Conceptually equivalent to the corresponding 'inverse' method accepting a
     * vector of wavelet coefficients.
     *
     * @param activations Collection of activations for all layers/levels in the 
              wavenet.
     * @returns Vector of values in position space.
     */
    arma::Col<double> inverse (const arma::field< arma::Col<double> >& activations);
    /**
     * @brief Inverse transform of wavelet coefficients.
     *
     * Conceptually equivalent to the corresponding 'inverse' method accepting a
     * collection of neural network activations.
     *
     * @param activations Vector of wavelet coefficients.
     * @returns Vector of values in position space.
     */
    arma::Col<double> inverse (const arma::Col<double>& y);

    /// 2D wavenet transform method(s).
    std::vector< std::vector< arma::field< arma::Col<double> > > > forward (const arma::Mat<double>& X);
    arma::Mat<double>                                              inverse (const arma::Mat<double>& Y);

    
    /// High-level cost method(s).
    
    double cost (const arma::Col<double>& y);
    double cost (const arma::Mat<double>& Y);
    
    arma::field< arma::Mat<double> > costMap (const arma::Mat<double>& X, const double& range, const unsigned& Ndiv);
    arma::field< arma::Mat<double> > costMap (const std::vector< arma::Mat<double> >& X, const double& range, const unsigned& Ndiv);
    
    // High-level basis method(s).
    arma::Mat<double> basisFunction1D (const unsigned& nRows, const unsigned& irow);
    arma::Mat<double> basisFunction2D (const unsigned& nRows, const unsigned& nCols, const unsigned& irow, const unsigned& icol);
    arma::Mat<double> basisFunction   (const unsigned& nRows, const unsigned& nCols, const unsigned& irow, const unsigned& icol);
    

public: 
    
    // Low-level learning method(s).
    void addMomentum   (const arma::Col<double>& momentum);
    void scaleMomentum (const double& factor);

    void clear ();
    void update (const arma::Col<double>& gradient);
    
    void cacheOperators       (const unsigned& m);
    void clearCachedOperators ();
    
    void cacheWeights       (const unsigned& m);
    void clearCachedWeights ();

    arma::Col<double> lowpassfilter      (const arma::Col<double>& x);
    arma::Col<double> highpassfilter     (const arma::Col<double>& x);
    arma::Col<double> inv_lowpassfilter  (const arma::Col<double>& y);
    arma::Col<double> inv_highpassfilter (const arma::Col<double>& y);

    const arma::Mat<double>& lowpassweight  (const unsigned& level, const unsigned& filt);
    const arma::Mat<double>& highpassweight (const unsigned& level, const unsigned& filt);

    arma::field< arma::Col<double> > backpropagate (const arma::Col<double>& delta, arma::field< arma::Col<double> > activations);
    
    void batchTrain (arma::Mat<double> X);
    void flushBatchQueue ();
    
    // Miscellaneous.
    arma::Col<double> coeffsFromActivations (const arma::field< arma::Col<double> >& activations);
    arma::Mat<double> coeffsFromActivations (const std::vector< std::vector< arma::field< arma::Col<double> > > >& Activations);
    

protected:

    /// Streaming operator(s).
    /**
     * As these methods relate chiefly to the Snapshot class, they are 
     * implemented in Snapshot.cxx.
     */
    friend       Snapshot& operator<< (      Snapshot& snap, const Wavenet& wavenet);
    friend const Snapshot& operator>> (const Snapshot& snap,       Wavenet& wavenet);


private: 
    
    double m_lambda  = 10.0;
    double m_alpha   =  0.01;
    double m_inertia =  0.;
    double m_inertiaTimeScale = 0.;
    
    arma::Col<double> m_filter;
    arma::Col<double> m_momentum;

    bool m_hasCachedOperators = false;
    arma::field< arma::Mat<double> > m_cachedLowpassOperators;
    arma::field< arma::Mat<double> > m_cachedHighpassOperators;

    bool m_hasCachedWeights = false;
    arma::field< arma::Mat<double> > m_cachedLowpassWeights;
    arma::field< arma::Mat<double> > m_cachedHighpassWeights;
    
    unsigned m_batchSize = 1;
    std::vector< arma::Col<double> > m_batchQueue;
    std::vector< arma::Col<double> > m_filterLog;
    std::vector< double >            m_costLog;
    
    bool m_wavelet = true;
    
};

/// Utility function(s).
// Frobenius inner product between matrices A and B. In the current design, this is the most significant performance bottleneck.
template<class T>
inline T frobeniusProduct (const arma::Mat<T>& A, const arma::Mat<T>& B) { 
    // Other, slower variants:
    //   return arma::trace( A * B.t() ); 
    //   return arma::as_scalar( arma::vectorise(A).t() * arma::vectorise(B) ); 
    return arma::accu( A % B );
};

// Vector outer product between vectors a and b.
template<class T>
inline arma::Mat<T> outerProduct (const arma::Col<T>& a, const arma::Col<T>& b) { 
    return (a * b.t()); 
}

} // namespace

#endif // WAVENET_WAVENET_H
