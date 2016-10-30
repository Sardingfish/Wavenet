#include "Wavenet/WaveletML.h"

 // Set method(s).
// -------------------------------------------------------------------

bool WaveletML::setLambda (const double& lambda) {
    _lambda = lambda;
    return true;
}

bool WaveletML::setAlpha (const double& alpha) {
    _alpha = alpha;
    return true;
}

bool WaveletML::setInertia (const double& inertia) {
    _inertia = inertia;
    return true;
}

bool WaveletML::setFilter (const arma::Col<double>& filter) {
    if (filter.is_empty()) {
        std::cout << "<WaveletML::setFilter> WARNING: Input filter is empty." << std::endl;
        return false;
    }
    if (filter.n_rows % 2) {
        std::cout << "<WaveletML::setFilter> WARNING: Input filter length is not a multiple of 2." << std::endl;
        return false;
    }
    _filter = filter;
    _filterLog.push_back(_filter);
    //_momentum.zeros(size(_filter));
    clearCachedOperators();
    //clearCachedWeights();
    
    // RMSprop
    if (_RMSprop.n_elem != _filter.n_elem) {
        _RMSprop.zeros(size(_filter));
    }
    if (_momentum.n_elem != _filter.n_elem) {
        _momentum.zeros(size(_filter));
    }
    
    return true;
}

bool WaveletML::setMomentum (const arma::Col<double>& momentum) {
    if (momentum.size() != _filter.size()) {
        std::cout << "<WaveletML::setMomentum> WARNING: Input momentum is not same size as stored filter." << std::endl;
        return false;
    }
    _momentum = momentum;
    return true;
}


bool WaveletML::setBatchSize (const unsigned& batchSize) {
    _batchSize = batchSize;
    return true;
}

bool WaveletML::doWavelet (const bool& wavelet) {
    _wavelet = wavelet;
    return true;
}

 // Print method(s).
// -------------------------------------------------------------------

void WaveletML::print () {
    printf("\n");
    printf("This WaveletML instance has the following properties:\n");
    printf("  lambda   : %4.1f  \n", _lambda);
    printf("  alpha    :  %4.2f \n", _alpha);
    printf("  inertia  :  %4.2f \n", _inertia);
    printf("  filter   : [");
    for (unsigned i = 0; i < _filter.n_elem; i++) {
        if (i > 0) { printf(", "); }
        printf("%6.3f", _filter(i));
    }
    printf("] \n");
    printf("  momentum : [");
    for (unsigned i = 0; i < _momentum.n_elem; i++) {
        if (i > 0) { printf(", "); }
        printf("%6.3f", _momentum(i));
    }
    printf("] \n");
    printf("  batch size : %u\n", _batchSize);
    printf("  batch queue : \n");
    for (unsigned j = 0; j < _batchQueue.size(); j++) {
        printf("    [");
        for (unsigned i = 0; i < _batchQueue.at(j).n_elem; i++) {
            if (i > 0) { printf(", "); }
            printf("%6.3f", _batchQueue.at(j)(i));
        }
        printf("] \n");
    }
    printf("\n");
    return;
}


// Storage methods(s).
// -------------------------------------------------------------------

void WaveletML::save (const string& filename) {
    Snapshot snap (filename);
    snap.save(this);
    return;
}

void WaveletML::load (const string& filename) {
    Snapshot snap (filename);
    snap.load(this);
    return;
}


 // High-level learning methods(s).
// -------------------------------------------------------------------

// Forward (1D).
arma::field< arma::Col<double> > WaveletML::forward (const arma::Col<double>& x) {

    unsigned m = log2(x.n_elem);
    
    arma::field< arma::Col<double> > activations(m + 1, 2);
    
    /**
     * Structure: 
     *   (m, 0) = Lowpass  coeffs. at level m
     *   (m, 1) = Highpass coeffs. at level m
    **/
    
    arma::Col<double> x_current = x;
    for (unsigned i = m; i --> 0; ) {
        activations(i, 0) = lowpassfilter (x_current);
        activations(i, 1) = highpassfilter(x_current);
        x_current = activations(i, 0);
    }
    
    activations(m, 0) = x;
    activations(m, 1) = zeros(size(x));
    
    return activations;
}

// Forward (2D) - returning activation fields.
arma::field< arma::field< arma::Col<double> > > WaveletML::forward (const arma::Mat<double>& X) {
    
    unsigned N = size(X,0);
    
    if (N != size(X, 1)) {
        std::cout << "<WaveletML::forward> WARNING: Input signal is not square." << std::endl;
        return arma::field< arma::field< arma::Col<double> > >();
    }
    
    arma::field< arma::field< arma::Col<double> > > Activations(N, 2); // { irow/icol; 0 = row, 1 = col}
    arma::Mat<double> Y = X;
    
    for (unsigned i1 = 0; i1 < N; i1++) {
        arma::Col<double> x = Y.row(i1).t();
        arma::field< arma::Col<double> > activations = forward(x);
        Activations(i1, 0) = activations;
        Y.row(i1) = coeffsFromActivations(activations).t();
    }
    
    for (unsigned i2 = 0; i2 < N; i2++) {
        arma::Col<double> x = Y.col(i2);
        arma::field< arma::Col<double> > activations = forward(x);
        Activations(i2, 1) = activations;
        Y.col(i2) = coeffsFromActivations(activations);
    }
    
    return Activations;
}

// Inverse (2D).
arma::Mat<double> WaveletML::inverse (const arma::Mat<double>& Y) {
    
    unsigned N1 = size(Y, 0);
    unsigned N2 = size(Y, 1);
    
    arma::Mat<double> X = Y;
    for (unsigned i2 = 0; i2 < N2; i2++) {
        arma::Col<double> x = X.col(i2);
        X.col(i2) = inverse(x);
    }

    for (unsigned i1 = 0; i1 < N1; i1++) {
        arma::Col<double> x = X.row(i1).t();
        X.row(i1) = inverse(x).t();
    }
    
    return X;
}

// Inverse (1D) – using activation fields.
arma::Col<double> WaveletML::inverse (const arma::field< arma::Col<double> >& activations) {
    
    unsigned m = log2(size(activations, 0));
    
    arma::Col<double> x = activations(0, 0);
    
    for (unsigned i = 0; i < m + 1; i++) {
        x  = inv_lowpassfilter (x);
        x += inv_highpassfilter(activations(i, 1));
    }
    
    return x;
}

// Inverse (1D) – using coefficients.
arma::Col<double> WaveletML::inverse (const arma::Col<double>& y) {
    unsigned m = log2(y.n_elem);
    
    arma::Col<double> x (1, fill::ones);
    x.fill(y(0));

    for (unsigned i = 0; i < m; i++) {
        x  = inv_lowpassfilter (x);
        x += inv_highpassfilter(y( span(pow(2, i), pow(2, i + 1) - 1) ));
    }
    
    return x;
}


 // High-level cost method(s).
// -------------------------------------------------------------------
double WaveletML::GiniCoeff (const arma::Col<double>& y) {
    unsigned N = y.n_elem;
    
    arma::Col<double> ySortAbs = sort(abs(y));
    arma::Row<double> indices  = linspace< arma::Row<double> >(1, N, N);
    
    double f = - dot( (2*indices - N - 1), ySortAbs);
    double g = N * sum(ySortAbs);
    
    return f/g + 1;
}

double WaveletML::GiniCoeff (const arma::Mat<double>& Y) {
    const arma::Col<double> y = vectorise(Y);
    return GiniCoeff(y);
}


arma::Col<double> WaveletML::GiniCoeffDeriv (const arma::Col<double>& y) {
   
    /**
     * Gini coefficient is written as: G({a}) = f({a})/g({a}) with :
     *   f({a}) = sum_i (2i - N - 1)|a_i|
     * for ordered a_i (a_i < a_{i+1}), and:
     *   g({a})  = N * sum_i |a_i|
     * Then: delta(i) = d/da_i G({a}) = f'*g - f*g' / g^2
    **/
        
    // Define some convenient variables.
    unsigned N = y.n_elem;
    arma::Col<double> yAbs  = abs(y);
    arma::Col<double> ySign = sign(y);
    uvec idxSortAbs = sort_index(yAbs, "ascend");
    arma::Col<double> ySortAbs = yAbs.elem(idxSortAbs);
    arma::Col<double> indices  = linspace< arma::Col<double> >(1, N, N);
    
    // Get numerator and denominator for Gini coefficient.
    double f = - dot( (2*indices - N - 1), ySortAbs );
    double g = N * sum(ySortAbs);
    
    // df/d|a| for sorted a.
    arma::Col<double> dfSort = - (2*indices - N - 1);
    
    // df/d|a| for a with original ordering.
    arma::Col<double> df (N, fill::zeros);
    df.elem(idxSortAbs) = dfSort;
    
    // df/da for a with original ordering.
    df = df % ySign; // Elementwise multiplication.
    
    // df/da for a with original ordering.
    arma::Col<double> dg = N * ySign;
    
    // Combined derivative.
    arma::Col<double> D = (df * g - f * dg)/pow(g,2.);
    
    return D;
}

arma::Mat<double> WaveletML::GiniCoeffDeriv (const arma::Mat<double>& Y) {
    arma::Col<double> y = vectorise(Y);
    arma::Col<double> D = GiniCoeffDeriv(y);
    return reshape(D, size(Y));
}

double WaveletML::SparseTerm (const arma::Col<double>& y) {
    return GiniCoeff(y);
}

double WaveletML::SparseTerm (const arma::Mat<double>& Y) {
    return GiniCoeff(Y);
}

double WaveletML::RegTerm    (const arma::Col<double>& a) {
    unsigned N = a.n_elem;
    arma::Col<double> kroenecker_delta (N+1, fill::zeros);
    kroenecker_delta(N/2) = 1;
    
    // C2
    arma::Mat<double> M (N+1, 3*N, fill::zeros);
    
    for (unsigned i = 0; i < N+1; i++) {
        M.submat(span(i,i), span(N,2*N-1)) = a.t();
        M.row(i) = rowshift(M.row(i), 2 * (i - N/2));
    }
    M = M.submat(span::all, span(N,2*N-1));
    
    double R = sum(square(M * a - kroenecker_delta));
    
    // Wavelet part.
    if (_wavelet) {
        double waveletTerm = 0.;
        arma::Col<double> b (N, fill::zeros);
        for (unsigned i = 0; i < N; i++) {
            b(i) = pow(-1, i) * a(N - i - 1);
        }
        
        // C1
        waveletTerm += sq(sum(a) - sqrt(2));

        // C3
        arma::Mat<double> M3 (N+1, 3*N, fill::zeros);
        for (unsigned i = 0; i < N+1; i++) {
            M3.submat(span(i,i), span(N,2*N-1)) = b.t();
            M3.row(i) = rowshift(M3.row(i), 2 * (i - N/2));
        }
        M3 = M3.submat(span::all, span(N,2*N-1));
        waveletTerm += sum(square(M3 * b - kroenecker_delta));
        
        // C4
        waveletTerm += sq(sum(b) - 0);

        // C5
        /* This should be automatically satisfied (confirm).
        arma::Mat<double> M5 (N+1, 3*N, fill::zeros);
        for (unsigned i = 0; i < N+1; i++) {
            M5.submat(span(i,i), span(N,2*N-1)) = b.t();
            M5.row(i) = rowshift(M5.row(i), 2 * (i - N/2));
        }
        M5 = M5.submat(span::all, span(N,2*N-1));
        waveletTerm += sum(square(M5 * a - kroenecker_delta));
         */
        
        // Add.
        R += waveletTerm;
    }
    
    return (_lambda/2.) * R;
}

arma::Col<double> WaveletML::SparseTermDeriv (const arma::Col<double>& y) {
    return GiniCoeffDeriv(y);
}

arma::Mat<double> WaveletML::SparseTermDeriv (const arma::Mat<double>& Y) {
    return GiniCoeffDeriv(Y);
}

arma::Col<double> WaveletML::RegTermDeriv    (const arma::Col<double>& a) {
    
    // Taking the derivative of each term in the sum of squared
    // deviations from the Kroeneker deltea, in the regularion term
    // of the cost function.
    //  - First term is the outer derivative.
    //  - Second term is the inner derivative.
    
    int N = a.n_elem;
    
    // -- Get outer derivative
    arma::Mat<double> M (N+1, 3*N, fill::zeros);
    
    for (unsigned i = 0; i < N+1; i++) {
        M.submat(span(i,i), span(N,2*N-1)) = a.t();
        M.row(i) = rowshift(M.row(i), 2 * (i - N/2));
    }
    M = M.submat(span::all, span(N,2*N-1));
    
    arma::Col<double> kroenecker_delta (N+1, fill::zeros);
    kroenecker_delta(N/2) = 1;
    
    arma::Col<double> outer = 2 * (M * a - kroenecker_delta);
    
    // -- Get inner derivative.
    arma::Mat<double> inner (N + 1, N, fill::zeros);
    
    for (int l = 0; l < N; l++) {
        for (int i = 0; i < N + 1; i++) {
            
            int d = 2 * abs(i - (int) N/2);
            if (l + d < N) {
                inner(i, l) += a(l + d);
            }
            if (l - d >= 0) {
                inner(i, l) += a(l - d);
            }
            
        }
    }
    
    arma::Col<double> D = (outer.t() * inner).t();
    
    // Wavelet part.
    if (_wavelet) {
        arma::Col<double> waveletTerm (N, fill::zeros);
        arma::Col<double> b     (N, fill::zeros);
        arma::Col<double> bsign (N, fill::zeros);
        for (unsigned i = 0; i < N; i++) {
            b    (i) = pow(-1, i) * a(N - i - 1);
            bsign(i) = pow(-1, N - i - 1);
        }
        arma::Col<double> kroenecker_delta (N+1, fill::zeros);
        kroenecker_delta(N/2) = 1;
        
        // C1
        waveletTerm += 2 * (sum(a) - sqrt(2)) * arma::Col<double> (N, fill::ones);
        
        // C3

        // -- Get outer derivative
        arma::Mat<double> M3 (N+1, 3*N, fill::zeros);
        for (unsigned i = 0; i < N+1; i++) {
            M3.submat(span(i,i), span(N,2*N-1)) = b.t();
            M3.row(i) = rowshift(M3.row(i), 2 * (i - N/2));
        }
        M3 = M3.submat(span::all, span(N,2*N-1));
        arma::Col<double> outer3 = 2 * (M3 * b - kroenecker_delta);
        
        // -- Get inner derivative.
        arma::Mat<double> inner3 (N + 1, N, fill::zeros);
        for (int l = 0; l < N; l++) {
            for (int i = 0; i < N + 1; i++) {
                int d = 2 * abs(i - (int) N/2);
                if (N - 1 - l + d < N) {
                    inner3(i, l) += pow(-1, N - 1 - l) * b(N - 1 - l + d);
                }
                if (N - 1 - l - d >= 0) {
                    inner3(i, l) += pow(-1, N - 1 - l) * b(N - 1 - l - d);
                }
            }
        }
        waveletTerm += (outer3.t() * inner3).t();
        

        // C4
        waveletTerm += 2 * (sum(b) - 0) * bsign;
        
        // C5
        /* This should be automatically satisfied (confirm).
        // -- Get outer derivative
        arma::Mat<double> M5 (N+1, 3*N, fill::zeros);
        for (unsigned i = 0; i < N+1; i++) {
            M5.submat(span(i,i), span(N,2*N-1)) = b.t();
            M5.row(i) = rowshift(M5.row(i), 2 * (i - N/2));
        }
        M5 = M5.submat(span::all, span(N,2*N-1));
        arma::Col<double> outer5 = 2 * (M * b - kroenecker_delta);
        
        // -- Get inner derivative.
        arma::Mat<double> inner5 (N + 1, N, fill::zeros);
        for (int l = 0; l < N; l++) {
            for (int i = 0; i < N + 1; i++) {
                int d = 2 * abs(i - (int) N/2);
                if (l + d < N) {
                    inner5(i, l) += b(l + d);
                }
                if (l - d >= 0) {
                    inner5(i, l) += b(l - d);
                }
            }
        }
        waveletTerm += (outer5.t() * inner5).t();
        */
        // Add.
        D += waveletTerm;
    }
    
    return (_lambda/2.) * D;
}

double WaveletML::cost (const arma::Col<double>& y) {
    // Sparsity term: Gini coefficient.
    double S = SparseTerm(y);
    
    // Regularisation term.
    double R = RegTerm(_filter);
    
    // Sum.
    double J = S + R;
    
    return J;
}

double WaveletML::cost (const arma::Mat<double>& Y) {
    arma::Col<double> y = vectorise(Y);
    return cost(y);
}

arma::field< arma::Mat<double> > WaveletML::costMap (const arma::Mat<double>& X, const double& range, const unsigned& Ndiv) {

    return costMap(vector< Mat<double> > ({X}), range, Ndiv);
}

arma::field< arma::Mat<double> > WaveletML::costMap (const std::vector< arma::Mat<double> >& X, const double& range, const unsigned& Ndiv) {
    
    arma::field< arma::Mat<double> > costs (3,1); // { J, S, R }
    costs.for_each( [Ndiv](arma::Mat<double>& m) { m.zeros(Ndiv, Ndiv); } );

    const unsigned nExample = X.size();
    
    std::cout << "Starting 'costMap' loop!"<< std::endl;
    for (unsigned i = 0; i < Ndiv; i++) {
        for (unsigned j = 0; j < Ndiv; j++) {
            std::cout << "  " << i << "/" << j << " out of " << Ndiv << std::endl;
            double a1 = (2*j/double(Ndiv - 1) - 1) * range;
            double a2 = (2*i/double(Ndiv - 1) - 1) * range;
            setFilter({a1, a2});
            for (unsigned iExample = 0; iExample < nExample; iExample++) {
                arma::field< arma::field< arma::Col<double> > > Activations = forward(X.at(iExample));
                arma::Mat<double> Y = coeffsFromActivations(Activations);
                costs(0,0).submat(span(i,i),span(j,j)) += cost(Y);
                costs(1,0).submat(span(i,i),span(j,j)) += SparseTerm(Y);
                costs(2,0).submat(span(i,i),span(j,j)) += RegTerm(_filter);
            }
        }
    }
    costs(0,0) /= (double) nExample;
    costs(1,0) /= (double) nExample;
    costs(2,0) /= (double) nExample;
    
    std::cout << "Done with 'costMap'!"<< std::endl;
    
    // ...
    
    return costs;
}


// High-level learnings method(s).
// -------------------------------------------------------------------

arma::Col<double> WaveletML::basisFct (const unsigned& N, const unsigned& i) {
    unsigned m = log2(N);
    if (!_hasCachedOperators || size(_cachedLowpassOperators, 0) < m) { cacheOperators(m - 1); }
    arma::Col<double> y (N, fill::zeros);
    y(i) = 1.;
    return inverse(y);
}

arma::Mat<double> WaveletML::basisFct (const unsigned& N, const unsigned& i, const unsigned& j) {
    unsigned m = log2(N);
    if (!_hasCachedOperators || size(_cachedLowpassOperators, 0) < m) { cacheOperators(m - 1); }
    arma::Mat<double> Y (N, N, fill::zeros);
    Y(i, j) = 1.;
    return inverse(Y);
}



TGraph WaveletML::getCostGraph (const std::vector< double >& costLog) {
    
    const unsigned N = costLog.size();
    double x[N], y[N];
    for (unsigned i = 0; i < N; i++) {
        x[i] = i;
        y[i] = costLog.at(i);
        
    }
    
    TGraph graph (N, x, y);
    
    return graph;
}


TGraph WaveletML::getCostGraph (const std::vector< arma::Col<double> >& filterLog, const std::vector< arma::Mat<double> >& X) {
    
    const unsigned N = filterLog.size();
    double x[N], y[N];
    for (unsigned i = 0; i < N; i++) {
        x[i] = i;
        y[i] = 0;
        setFilter(filterLog.at(i));
        for (const arma::Mat<double>& x : X) {
            y[i] += cost(coeffsFromActivations(forward(x)));
        }
        y[i] /= (double) X.size();
        
    }
    
    TGraph graph (N, x, y);
    
    return graph;
}



 // Low-level learnings method(s).
// -------------------------------------------------------------------

void WaveletML::addMomentum   (const arma::Col<double>& momentum) {
    if (_momentum.n_elem > 0) {
        _momentum += momentum;
    } else {
        _momentum  = momentum;
    }
    return;
}

void WaveletML::scaleMomentum (const double& factor) {
    _momentum = _momentum * factor;
    return;
}
void WaveletML::clear () {
    scaleMomentum(0.);
    clearFilterLog();
    clearCostLog();
    clearCachedOperators();
    return;
}

void WaveletML::update (const arma::Col<double>& gradient) {
    
    // RMSprop (somehow _extremely_ time consuming...)
    //_RMSprop = _gamma * _RMSprop + (1.0 - _gamma) * square(gradient);
    //arma::Col<double> sqrtR = sqrt(_RMSprop);
    //arma::Col<double> newGradient = accu(sqrtR) * (gradient / sqrtR);  // Normalise to unity?
    //arma::Col<double> newGradient = gradient % pow(_RMSprop, -0.5);
    
    // Update.
    scaleMomentum( _inertia );
    //addMomentum( - _alpha * newGradient); // ... * gradient);
    addMomentum( - _alpha * gradient);
    setFilter( _filter + _momentum );
    return;
}

void WaveletML::cacheOperators (const unsigned& m) {
    
    clearCachedOperators();

    _cachedLowpassOperators .set_size(m + 1, 1);
    _cachedHighpassOperators.set_size(m + 1, 1);

    for (unsigned i = 0; i <= m; i++) {
        _cachedLowpassOperators (i, 0) = LowpassOperator (_filter, i);
        _cachedHighpassOperators(i, 0) = HighpassOperator(_filter, i);
    }
    
    _hasCachedOperators = true;
    return;
}

void WaveletML::clearCachedOperators () {
    _cachedLowpassOperators .reset();
    _cachedHighpassOperators.reset();
    _hasCachedOperators = false;
    return;
}

void WaveletML::cacheWeights (const unsigned& m) {
    
    std::cout << "Caching matrix weights (" << m << ")." << std::endl;
    clearCachedWeights();
    if (!_hasCachedOperators) { cacheOperators(m); }
    
    const unsigned N = _filter.n_elem;
    
    _cachedLowpassWeights .set_size(m + 1, N);
    _cachedHighpassWeights.set_size(m + 1, N);
    
    for (unsigned i = 0; i <= m; i++) {
    
        arma::Col<double> t (N, fill::zeros);
        
        for (unsigned k = 0; k < N; k++) {
            t.fill(0);
            t(k) = 1;
            _cachedHighpassWeights(i, k) = HighpassOperator(t, i);
            _cachedLowpassWeights (i, k) = LowpassOperator (t, i);
        }
        
    }
    
    _hasCachedWeights = true;
    return;
}

void WaveletML::clearCachedWeights () {
    _cachedLowpassWeights .reset();
    _cachedHighpassWeights.reset();
    _hasCachedWeights = false;
    return;
}

arma::Col<double> WaveletML::lowpassfilter      (const arma::Col<double>& x) {
    unsigned m = log2(x.n_elem);
    if (!_hasCachedOperators || size(_cachedLowpassOperators, 0) < m) { cacheOperators(m); }
    assert(m > 0);
    return _cachedLowpassOperators(m - 1, 0) * x;
}

arma::Col<double> WaveletML::highpassfilter     (const arma::Col<double>& x) {
    unsigned m = log2(x.n_elem);
    if (!_hasCachedOperators || size(_cachedHighpassOperators, 0) < m) { cacheOperators(m); }
    assert(m > 0);
    return _cachedHighpassOperators(m - 1, 0) * x;
}

arma::Col<double> WaveletML::inv_lowpassfilter  (const arma::Col<double>& y) {
    unsigned m = log2(y.n_elem);
    if (!_hasCachedOperators || size(_cachedLowpassOperators, 0) <= m) { cacheOperators(m); }
    return _cachedLowpassOperators(m, 0).t() * y;
}

arma::Col<double> WaveletML::inv_highpassfilter (const arma::Col<double>& y) {
    unsigned m = log2(y.n_elem);
    if (!_hasCachedOperators || size(_cachedHighpassOperators, 0) <= m) { cacheOperators(m); }
    return _cachedHighpassOperators(m, 0).t() * y;
}

const arma::Mat<double>& WaveletML::lowpassweight (const unsigned& level, const unsigned& filt) {
    if (!_hasCachedWeights || size(_cachedLowpassWeights, 0) <= level) { cacheWeights(level); }
    return _cachedLowpassWeights(level, filt);
}

const arma::Mat<double>& WaveletML::highpassweight (const unsigned& level, const unsigned& filt) {
    if (!_hasCachedWeights || size(_cachedHighpassWeights, 0) <= level) { cacheWeights(level); }
    return _cachedHighpassWeights(level, filt);
}

arma::field< arma::Col<double> > WaveletML::ComputeDelta (const arma::Col<double>& delta, arma::field< arma::Col<double> > activations) {
    
    unsigned N = _filter.n_elem;
    unsigned m = size(activations, 0) - 1;
    
    if (!_hasCachedWeights) { cacheWeights(m); }
    
    arma::Col<double> Delta  (N, fill::zeros);
    
    arma::Col<double> delta_LP (1), delta_HP (1), delta_new_LP, delta_new_HP, activ_LP;
    delta_LP.row(0) = delta.row(0);
    delta_HP.row(0) = delta.row(1);
    
    for (unsigned i = 0; i < m; i++) {
        
        activ_LP = activations(i + 1, 0);
        
        // * Lowpass
        arma::Mat<double> error_LP = delta_LP * activ_LP.t();
       
        for (unsigned k = 0; k < N; k++) {
            /*arma::Mat<double> weighted_errors = std::move(lowpassweight (i, k) % error_LP); // Elementwise multiplication
            Delta (k) += accu(weighted_errors);
             */
            Delta (k) += trace( lowpassweight (i, k) * error_LP.t() );
        }
        
        // * Highpass
        arma::Mat<double> error_HP = delta_HP * activ_LP.t();
      
        for (unsigned k = 0; k < N; k++) {
            /*
            arma::Mat<double> weighted_errors = std::move(highpassweight (i, k)  % error_HP); // Elementwise multiplication
            Delta(k)  += accu(weighted_errors);
             */
            Delta (k) += trace( highpassweight (i, k) * error_HP.t() );
        }

        // * Next level.
        delta_LP = std::move(inv_lowpassfilter(delta_LP) + inv_highpassfilter(delta_HP));
        if (i != m - 1) {
            delta_HP = delta(span( pow(2, i + 1), pow(2, i + 2) - 1 ));
        } else {
            delta_HP.copy_size(delta_LP);
            delta_HP.zeros();
        }
        
    }
    
    arma::field< arma::Col<double> > outField(2,1);
    
    outField(0,0) = Delta;
    outField(1,0) = delta_LP;
    
    return outField;
}

void WaveletML::batchTrain (arma::Mat<double> X) {
    
    unsigned N = size(X, 0);
    
    arma::field< arma::field< arma::Col<double> > > Activations = forward(X);
    
    arma::Mat< double > Y (size(X));
    for (unsigned i = 0; i < N; i++) {
        Y.col(i) = coeffsFromActivations( Activations(i,1) );
    }
    
    arma::Mat< double > delta = SparseTermDeriv(Y), new_delta(size(delta)); 
    
    arma::Col<double> Delta  (size(_filter), fill::zeros);
    arma::Col<double> weight (size(_filter), fill::zeros);
    for (unsigned i = 0; i < N; i++) {
        arma::field< arma::Col<double> > outField = ComputeDelta(delta.col(i), Activations(i,1));
        Delta += outField(0,0);
        new_delta.col(i) = outField(1,0);
    }
    
    for (unsigned i = 0; i < N; i++) {
        arma::field< arma::Col<double> > outField = ComputeDelta(new_delta.row(i).t(), Activations(i,0));
        Delta += outField(0,0);
    }
    
    arma::Col<double> regularisation = RegTermDeriv(_filter);
    
    _batchQueue.push_back(Delta + regularisation);
    _costLog.back() += cost(Y);
    if (_batchQueue.size() >= _batchSize) { flushBatchQueue(); }
    return;
    
}


void WaveletML::flushBatchQueue () {
    
    if (!_batchQueue.size()) { return; }

    arma::Col<double> gradient (size(_filter), fill::zeros);
    
    for (unsigned i = 0; i < _batchQueue.size(); i++) {
        gradient += _batchQueue.at(i);
    }
    gradient /= float(_batchQueue.size());
    
    this->update(gradient);

    _costLog.back() /= float(_batchQueue.size());
    _costLog.push_back(0);
    
    _batchQueue.clear();
    
    return;
}


 // Miscellaneous.
// -------------------------------------------------------------------
arma::Col<double> WaveletML::coeffsFromActivations (const arma::field< arma::Col<double> >& activations) {
    
    unsigned m = size(activations, 0) - 1;
    
    arma::Col<double> y (pow(2, m), fill::zeros);

    y(span(0,0)) = activations(0, 0);
    for (unsigned i = 0; i < m; i++) {
        y( span(pow(2, i), pow(2, i + 1) - 1) ) = activations(i, 1);
    }
    
    return y;
    
}

arma::Mat<double> WaveletML::coeffsFromActivations (const arma::field< arma::field< arma::Col<double> > >& Activations) {
    
    unsigned N = size(Activations, 0); // nRows or nCols
    
    arma::Mat<double> Y (N, N, fill::zeros);
    
    for (unsigned i = 0; i < N; i++) {
        Y.col(i) = coeffsFromActivations(Activations(i, 1));
    }
    
    return Y;
    
}

