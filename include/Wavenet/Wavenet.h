#ifndef WAVENET_WAVENET_H
#define WAVENET_WAVENET_H

/**
 * @file Wavenet.h
 * @author Andreas Sogaard
**/

// STL include(s).
#include <fstream>
#include <iostream>
#include <stdio.h> /* printf */
#include <vector> /* std::vector */
#include <string> /* std::string */
#include <cmath> /* log2 */
#include <assert.h> /* assert */
#include <stdlib.h> /* system */
#include <utility> /* std::move */
#include <algorithm> /* std::max */

// ROOT include(s).
/* @TODO: Make dependent on installation. */
#include "TGraph.h" 

// Armadillo include(s).
#include <armadillo>

// Wavenet include(s).
#include "Wavenet/Utils.h"
#include "Wavenet/Logger.h"
#include "Wavenet/LowpassOperator.h"
#include "Wavenet/HighpassOperator.h"
#include "Wavenet/Snapshot.h"


namespace Wavenet {

class Wavenet : Logger {
    
    friend class Snapshot;
    
public:
    
    // Constructor(s).
    Wavenet () {};
    Wavenet (const double& lambda) :
        _lambda(lambda)
    {};
    Wavenet (const double& lambda, const double& alpha) :
        _lambda(lambda), _alpha(alpha)
    {};
    Wavenet (const double& lambda, const double& alpha, const double& inertia) :
        _lambda(lambda), _alpha(alpha), _inertia(inertia)
    {};
    Wavenet (const Wavenet& other) :
        _lambda(other._lambda), _alpha(other._alpha), _inertia(other._inertia), _filter(other._filter)
    {};
    
    // Destructor.
    ~Wavenet () {};
    
    // Get method(s).
    inline double lambda ()  { return _lambda; }
    inline double alpha ()   { return _alpha; }
    inline double inertia () { return _inertia; }
    inline double inertiaTimeScale () { return _inertiaTimeScale; }
    
    inline arma::Col<double> filter ()   { return _filter; }
    inline arma::Col<double> momentum () { return _momentum; }

    inline int    batchSize () { return _batchSize; }

    inline std::vector< arma::Col<double> >    filterLog () { return _filterLog; }
    inline void                           clearFilterLog () { return _filterLog.clear(); }

    inline std::vector< double > getCostLog () { return _costLog; }
    inline std::vector< double >    costLog () { return getCostLog(); }
    inline void                clearCostLog () { return _costLog.resize(1, 0); }

    
    // Set method(s).
    bool setLambda           (const double& lambda);
    bool setAlpha            (const double& alpha);
    bool setInertia          (const double& inertia);
    bool setInertiaTimeScale (const double& inertiaTimeScale);
    bool setFilter           (const arma::Col<double>& filter);
    bool setMomentum         (const arma::Col<double>& momentum);
    bool setBatchSize        (const unsigned& batchSize);
    bool doWavelet           (const bool& wavelet);
    
    // Print method(s).
    void print ();

    // Storage method(s).
    inline void saveAs (const std::string& filename) { save(filename); return; };
           void save   (const std::string& filename);
           void load   (const std::string& filename);
    
    // High-level learning methods(s).
    // -- 1D.
    arma::field< arma::Col<double> > forward (const arma::Col<double>& x);
    
    arma::Col<double>                inverse (const arma::field< arma::Col<double> >& activations);
    arma::Col<double>                inverse (const arma::Col<double>& y);

    // -- 2D.
    std::vector< std::vector< arma::field< arma::Col<double> > > > forward (const arma::Mat<double>& X);
    arma::Mat<double>                                              inverse (const arma::Mat<double>& Y);

    
    // High-level cost method(s).
    double GiniCoeff (const arma::Col<double>& y);
    double GiniCoeff (const arma::Mat<double>& Y);

    arma::Col<double> GiniCoeffDeriv (const arma::Col<double>& y);
    arma::Mat<double> GiniCoeffDeriv (const arma::Mat<double>& Y);

    double SparseTerm (const arma::Col<double>& y);
    double SparseTerm (const arma::Mat<double>& Y);
    double RegTerm    (const arma::Col<double>& y);

    arma::Col<double> SparseTermDeriv (const arma::Col<double>& y);
    arma::Mat<double> SparseTermDeriv (const arma::Mat<double>& Y);
    arma::Col<double> RegTermDeriv    (const arma::Col<double>& y);

    double cost (const arma::Col<double>& y);
    double cost (const arma::Mat<double>& Y);
    
    arma::field< arma::Mat<double> > costMap (const arma::Mat<double>& X, const double& range, const unsigned& Ndiv);
    arma::field< arma::Mat<double> > costMap (const std::vector< arma::Mat<double> >& X, const double& range, const unsigned& Ndiv);
    
    // High-level basis method(s).
    arma::Col<double> basisFct (const unsigned& nRows, const unsigned& irow);
    arma::Mat<double> basisFct (const unsigned& nRows, const unsigned& nCols, const unsigned& irow, const unsigned& icol);
    
    TGraph getCostGraph (const std::vector< double >& costLog);
    TGraph getCostGraph (const std::vector< arma::Col<double> >& filterLog, const std::vector< arma::Mat<double> >& X);
    

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

    arma::field< arma::Col<double> >  ComputeDelta (const arma::Col<double>& delta, arma::field< arma::Col<double> > activations);
    
    void batchTrain (arma::Mat<double> X);
    void flushBatchQueue ();
    
    // Miscellaneous.
    arma::Col<double> coeffsFromActivations (const arma::field< arma::Col<double> >& activations);
    arma::Mat<double> coeffsFromActivations (const std::vector< std::vector< arma::field< arma::Col<double> > > >& Activations);
    

private: 
    
    double _lambda  = 10.0;
    double _alpha   =  0.05;
    double _inertia =  0.;
    double _inertiaTimeScale = 0.;
    
    arma::Col<double> _filter;
    arma::Col<double> _momentum;

    bool _hasCachedOperators = false;
    arma::field< arma::Mat<double> > _cachedLowpassOperators;
    arma::field< arma::Mat<double> > _cachedHighpassOperators;

    bool _hasCachedWeights = false;
    arma::field< arma::Mat<double> > _cachedLowpassWeights;
    arma::field< arma::Mat<double> > _cachedHighpassWeights;

    unsigned _batchSize = 1;
    std::vector< arma::Col<double> > _batchQueue;
    std::vector< arma::Col<double> > _filterLog;
    std::vector< double >            _costLog;
    
    bool _wavelet = true;
    
    // RMSprop stuff.
    double _gamma = 0.1;
    arma::Col<double> _RMSprop;
    arma::Mat<double> _AdaGrad;
    
};

} // namespace

#endif // WAVENET_WAVENET_H
