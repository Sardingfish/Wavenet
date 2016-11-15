#include "Wavenet/Wavenet.h"

namespace wavenet {
    
// Set method(s).
// -------------------------------------------------------------------

bool Wavenet::setLambda (const double& lambda) {
    m_lambda = lambda;
    return true;
}

bool Wavenet::setAlpha (const double& alpha) {
    m_alpha = alpha;
    return true;
}

bool Wavenet::setInertia (const double& inertia) {
    m_inertia = inertia;
    return true;
}

bool Wavenet::setInertiaTimeScale (const double& inertiaTimeScale) {
    if (inertiaTimeScale < 0.) {
        WARNING("Inertia time scale has to be positive. Received value %f. Exiting.", inertiaTimeScale);
        return false;
    }
    m_inertiaTimeScale = inertiaTimeScale;
    return true;
}

bool Wavenet::setFilter (const arma::Col<double>& filter) {
    if (filter.is_empty()) {
        WARNING("Input filter is empty.");
        return false;
    }
    if (filter.n_rows % 2) {
        WARNING("Input filter length is not a multiple of 2.");
        return false;
    }
    m_filter = filter;
    m_filterLog.push_back(m_filter);
    clearCachedOperators();
    
    // If the filter size is changes, resize the momentum vector accordingly.
    if (m_momentum.n_elem != m_filter.n_elem) {
        m_momentum.zeros(size(m_filter));
    }
    
    return true;
}

bool Wavenet::setMomentum (const arma::Col<double>& momentum) {
    if (momentum.size() != m_filter.size()) {
        WARNING("Input momentum is not same size as stored filter.");
        return false;
    }
    m_momentum = momentum;
    return true;
}


bool Wavenet::setBatchSize (const unsigned& batchSize) {
    m_batchSize = batchSize;
    return true;
}

bool Wavenet::doWavelet (const bool& wavelet) {
    m_wavelet = wavelet;
    return true;
}

 // Print method(s).
// -------------------------------------------------------------------

void Wavenet::print () const {
    INFO("");
    INFO("- - - - - - - - - - - - - - - - - - - - - - - - - -");
    INFO("This Wavenet instance has the following properties:");
    INFO("  lambda            : %4.1f  ", m_lambda);
    INFO("  alpha             :  %4.2f ", m_alpha);
    INFO("  inertia           :  %4.2f ", m_inertia);
    INFO("  inertiaTimeScale  :  %4.2f ", m_inertiaTimeScale);

    // Filter coefficients.
    std::string filterString = "";
    for (unsigned i = 0; i < m_filter.n_elem; i++) {
        if (i > 0) { filterString += ", "; }
        filterString += std::to_string(m_filter(i));
    }
    INFO("  filter   : [%s]", filterString.c_str());

    // Momentum.
    std::string momentumString = "";
    for (unsigned i = 0; i < m_momentum.n_elem; i++) {
        if (i > 0) { momentumString += ", "; }
        momentumString += std::to_string(m_momentum(i));
    }
    INFO("  momentum : [%s]", momentumString.c_str());
    
    
    // Batch queue:
    INFO("  batch size : %u", m_batchSize);
    INFO("  batch queue : %s", (m_batchQueue.size() == 0 ? "(empty)" : ""));
    for (unsigned j = 0; j < m_batchQueue.size(); j++) {
        std::string batchString = "";
        for (unsigned i = 0; i < m_batchQueue.at(j).n_elem; i++) {
            if (i > 0) { batchString += ", "; }
            batchString += std::to_string(m_batchQueue.at(j)(i));
        }
        INFO("         : [%s]", batchString.c_str());
    }
    INFO("- - - - - - - - - - - - - - - - - - - - - - - - - -");
    INFO("");
    return;
}


// Storage methods(s).
// -------------------------------------------------------------------
void Wavenet::save (Snapshot snap) {

    DEBUG("Saving snapshot '%s'.", snap.file().c_str());

    // Perform checks.
    if (strcmp(snap.file().substr(0,1).c_str(), "/") == 0) {
        FCTWARNING("File '%s' not accepted. Only accepting realtive paths.", snap.file().c_str());
        return;
    }
    
    if (snap.exists()) {
        DEBUG("File '%s' already exists. Overwriting.", snap.file().c_str()); 
    }
    
    if (snap.file().find("/") != std::string::npos) {
        std::string dir = snap.file().substr(0,snap.file().find_last_of("/"));
        if (!dirExists(dir)) {
            FCTWARNING("Directory '%s' does not exist. Creating it.", dir.c_str());
            system(("mkdir -p " + dir).c_str());
        }
    }

    // Stream the instance to file.
    snap << *this;
    return;
}

void Wavenet::load (Snapshot snap) {

    DEBUG("Loading snapshot '%s'.", snap.file().c_str())

    // Perform checks.
    if (!fileExists(snap.file())) {
        FCTWARNING("File '%s' doesn't exists.", snap.file().c_str());
        return;
    }
    
    // Stream the instance from file.
    snap >> *this;
    return;
}


 // High-level learning methods(s).
// -------------------------------------------------------------------

// Forward (1D).
arma::field< arma::Col<double> > Wavenet::forward (const arma::Col<double>& x) {

    const unsigned m = log2(x.n_elem);
    
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
std::vector< std::vector< arma::field< arma::Col<double> > > > Wavenet::forward (const arma::Mat<double>& X) {
    
    const unsigned nRows = size(X,0); // Rows.
    const unsigned nCols = size(X,1); // Columns.
    
    if (nRows != nCols) {
        VERBOSE("Input signal is not square.");
    }
    
    std::vector< std::vector< arma::field< arma::Col<double> > > > Activations(2);
    Activations.at(0).resize(nRows);
    Activations.at(1).resize(nCols);

    arma::Mat<double> Y = X;

    // Rows.
    for (unsigned irow = 0; irow < nRows; irow++) {
        arma::Col<double> x = Y.row(irow).t();
        arma::field< arma::Col<double> > activations = forward(x);
        Y.row(irow) = coeffsFromActivations(activations).t();
        Activations.at(0).at(irow) = std::move(activations);
    }
    
    // Columns.
    for (unsigned icol = 0; icol < nCols; icol++) {
        arma::Col<double> x = Y.col(icol);
        arma::field< arma::Col<double> > activations = forward(x);
        Y.col(icol) = coeffsFromActivations(activations);
        Activations.at(1).at(icol) = std::move(activations);
    }
    
    return Activations;
}

// Inverse (2D).
arma::Mat<double> Wavenet::inverse (const arma::Mat<double>& Y) {
    
    const unsigned N1 = size(Y, 0);
    const unsigned N2 = size(Y, 1);
    
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
arma::Col<double> Wavenet::inverse (const arma::field< arma::Col<double> >& activations) {
    
    const unsigned m = log2(size(activations, 0));
    
    arma::Col<double> x = activations(0, 0);
    
    for (unsigned i = 0; i < m + 1; i++) {
        x  = inv_lowpassfilter (x);
        x += inv_highpassfilter(activations(i, 1));
    }
    
    return x;
}

// Inverse (1D) – using coefficients.
arma::Col<double> Wavenet::inverse (const arma::Col<double>& y) {

    const unsigned m = log2(y.n_elem);
    
    arma::Col<double> x (1, arma::fill::ones);
    x.fill(y(0));

    for (unsigned i = 0; i < m; i++) {
        x  = inv_lowpassfilter (x);
        x += inv_highpassfilter(y( arma::span(pow(2, i), pow(2, i + 1) - 1) ));
    }
    
    return x;
}


 // High-level cost method(s).
// -------------------------------------------------------------------
double Wavenet::GiniCoeff (const arma::Col<double>& y) {
    
    const int N = y.n_elem;
    
    arma::Col<double> ySortAbs = sort(abs(y));
    arma::Row<double> indices  = arma::linspace< arma::Row<double> >(1, N, N);
    
    double f = - dot( (2*indices - N - 1), ySortAbs);
    double g = N * sum(ySortAbs);
    
    return f/g + 1;
}

double Wavenet::GiniCoeff (const arma::Mat<double>& Y) {
    const arma::Col<double> y = vectorise(Y);
    return GiniCoeff(y);
}


arma::Col<double> Wavenet::GiniCoeffDeriv (const arma::Col<double>& y) {
   
    /**
     * Gini coefficient is written as: G({a}) = f({a})/g({a}) with :
     *   f({a}) = sum_i (2i - N - 1)|a_i|
     * for ordered a_i (a_i < a_{i+1}), and:
     *   g({a})  = N * sum_i |a_i|
     * Then: delta(i) = d/da_i G({a}) = f'*g - f*g' / g^2
    **/
        
    // Define some convenient variables.
    const int N = y.n_elem;
    arma::Col<double> yAbs  = abs(y);
    arma::Col<double> ySign = sign(y);
    arma::uvec idxSortAbs = sort_index(yAbs, "ascend");
    arma::Col<double> ySortAbs = yAbs.elem(idxSortAbs);
    arma::Col<double> indices  = arma::linspace< arma::Col<double> >(1, N, N);
    
    // Get numerator and denominator for Gini coefficient.
    double f = - dot( (2*indices - N - 1), ySortAbs );
    double g = N * sum(ySortAbs);
    
    // df/d|a| for sorted a.
    arma::Col<double> dfSort = - (2*indices - N - 1);
    
    // df/d|a| for a with original ordering.
    arma::Col<double> df (N, arma::fill::zeros);
    df.elem(idxSortAbs) = dfSort;
    
    // df/da for a with original ordering.
    df = df % ySign; // Elementwise multiplication.
    
    // df/da for a with original ordering.
    arma::Col<double> dg = N * ySign;
    
    // Combined derivative.
    arma::Col<double> D = (df * g - f * dg)/pow(g,2.);
    
    return D;
}

arma::Mat<double> Wavenet::GiniCoeffDeriv (const arma::Mat<double>& Y) {
    arma::Col<double> y = vectorise(Y);
    arma::Col<double> D = GiniCoeffDeriv(y);
    return reshape(D, size(Y));
}

double Wavenet::SparseTerm (const arma::Col<double>& y) {
    return GiniCoeff(y);
}

double Wavenet::SparseTerm (const arma::Mat<double>& Y) {
    return GiniCoeff(Y);
}

double Wavenet::RegTerm    (const arma::Col<double>& a) {
    
    const int N = a.n_elem;
    arma::Col<double> kroenecker_delta (N+1, arma::fill::zeros);
    kroenecker_delta(N/2) = 1;
    
    // Regularisation term.
    double R = 0.; 

    /**
     * (C2): Orthogonality of scaling functions
     *
     * Mathematical condition:
     *   \sum_{k} a_{k} a_{k + 2m} = #delta_{0,m} \quad \forall m \in \mathbb{Z}
     *
     * Implementation:
     *   Let a be the vector of filter coefficients, of length N. 
     *
     *   Let N = 4. Then
     *
     *         |  0  0  0  0  0  0  0  0  0  0  0  0  |
     *         |  0  0  0  0  0  0  0  0  0  0  0  0  |
     *     M = |  0  0  0  0  0  0  0  0  0  0  0  0  |
     *         |  0  0  0  0  0  0  0  0  0  0  0  0  |
     *         |  0  0  0  0  0  0  0  0  0  0  0  0  |
     *
     *     For i = 0:
     *         |  0  0  0  0  a0 a1 a2 a3 0  0  0  0  |
     *         |  0  0  0  0  0  0  0  0  0  0  0  0  |
     *     M = |  0  0  0  0  0  0  0  0  0  0  0  0  |
     *         |  0  0  0  0  0  0  0  0  0  0  0  0  |
     *         |  0  0  0  0  0  0  0  0  0  0  0  0  |
     *
     *     (Shift by 2 * (0 - 4/2) = -4)
     *         |  0  0  0  0  0  0  0  0  a0 a1 a2 a3 |
     *         |  0  0  0  0  0  0  0  0  0  0  0  0  |
     *     M = |  0  0  0  0  0  0  0  0  0  0  0  0  |
     *         |  0  0  0  0  0  0  0  0  0  0  0  0  |
     *         |  0  0  0  0  0  0  0  0  0  0  0  0  |
     *
     *     For i = 1:
     *         |  0  0  0  0  0  0  0  0  a0 a1 a2 a3 |
     *         |  0  0  0  0  a0 a1 a2 a3 0  0  0  0  |
     *     M = |  0  0  0  0  0  0  0  0  0  0  0  0  |
     *         |  0  0  0  0  0  0  0  0  0  0  0  0  |
     *         |  0  0  0  0  0  0  0  0  0  0  0  0  |
     *
     *     (Shift by 2 * (1 - 4/2) = -2)
     *         |  0  0  0  0  0  0  0  0  a0 a1 a2 a3 |
     *         |  0  0  0  0  0  0  a0 a1 a2 a3 0  0  |
     *     M = |  0  0  0  0  0  0  0  0  0  0  0  0  |
     *         |  0  0  0  0  0  0  0  0  0  0  0  0  |
     *         |  0  0  0  0  0  0  0  0  0  0  0  0  |
     *
     *     For i = 2:
     *         |  0  0  0  0  0  0  0  0  a0 a1 a2 a3 |
     *         |  0  0  0  0  0  0  a0 a1 a2 a3 0  0  |
     *     M = |  0  0  0  0  a0 a1 a2 a3 0  0  0  0  |
     *         |  0  0  0  0  0  0  0  0  0  0  0  0  |
     *         |  0  0  0  0  0  0  0  0  0  0  0  0  |
     *
     *     (Shift by 2 * (2 - 4/2) = 0)
     *         |  0  0  0  0  0  0  0  0  a0 a1 a2 a3 |
     *         |  0  0  0  0  0  0  a0 a1 a2 a3 0  0  |
     *     M = |  0  0  0  0  a0 a1 a2 a3 0  0  0  0  |
     *         |  0  0  0  0  0  0  0  0  0  0  0  0  |
     *         |  0  0  0  0  0  0  0  0  0  0  0  0  |
     *
     *     (...)
     *
     *         |  0  0  0  0  0  0  0  0  a0 a1 a2 a3 |
     *         |  0  0  0  0  0  0  a0 a1 a2 a3 0  0  |
     *     M = |  0  0  0  0  a0 a1 a2 a3 0  0  0  0  |
     *         |  0  0  a0 a1 a2 a3 0  0  0  0  0  0  |
     *         |  a0 a1 a2 a3 0  0  0  0  0  0  0  0  |
     *
     *   Take sub-matrix
     *              |  0  0  0  0  \| 0  0  0  0  |\ a0 a1 a2 a3 |
     *              |  0  0  0  0  \| 0  0  a0 a1 |\ a2 a3 0  0  |
     *     M -> M = |  0  0  0  0  \| a0 a1 a2 a3 |\ 0  0  0  0  |
     *              |  0  0  a0 a1 \| a2 a3 0  0  |\ 0  0  0  0  |
     *              |  a0 a1 a2 a3 \| 0  0  0  0  |\ 0  0  0  0  |
     *
     *              |  0  0  0  0  |
     *              |  0  0  a0 a1 |
     *            = |  a0 a1 a2 a3 |
     *              |  a2 a3 0  0  |
     *              |  0  0  0  0  |
     *
     *   Finally, multiply
     *
     *             |  0  0  0  0  |   | a0 |   |         0         |    | 0 |
     *             |  0  0  a0 a1 |   | a1 |   | a0 * a2 + a1 * a3 |    | 0 |
     *     M * a = |  a0 a1 a2 a3 | * | a2 | = |        |a|^2      | == | 1 | = kroenecker_delta
     *             |  a2 a3 0  0  |   | a3 |   | a0 * a2 + a1 * a3 |    | 0 |
     *             |  0  0  0  0  |            |         0         |    | 0 |
     *
     */
    arma::Mat<double> M (N+1, 3*N, arma::fill::zeros);
    
    for (unsigned i = 0; i < N+1; i++) {
        M.submat(arma::span(i,i), arma::span(N,2*N-1)) = a.t();
        M.row(i) = rowshift(M.row(i), 2 * (N/2 - i));
    }
    M = M.submat(arma::span::all, arma::span(N,2*N-1));
    
    R += sum(square(M * a - kroenecker_delta));
    

    // Wavelet part.
    if (m_wavelet) {

        // Compute high-pass coefficients.
        arma::Col<double> b (N, arma::fill::zeros);
        for (unsigned i = 0; i < N; i++) {
            b(i) = pow(-1, i) * a(N - i - 1);
        }
        
        /**
         * (C1): Dilation equation.
         *
         * Mathmatical condition:
         *   \sum_{k} a_{k} = \sqrt{2}
         */
        R += sq(sum(a) - sqrt(2));


        /**
         * (C3): Orthonormality of wavelet functions.
         *
         * Mathematical condition:
         *   \sum_{k} b_{k} b_{k + 2m} = \delta_{0,m} \quad \forall m \in \mathbb{Z}
         */
        arma::Mat<double> M3 (N+1, 3*N, arma::fill::zeros);
        for (unsigned i = 0; i < N+1; i++) {
            M3.submat(arma::span(i,i), arma::span(N,2*N-1)) = b.t();
            M3.row(i) = rowshift(M3.row(i), 2 * (N/2 - i));
        }
        M3 = M3.submat(arma::span::all, arma::span(N,2*N-1));

        R += sum(square(M3 * b - kroenecker_delta));

        
        /**
         * (C4): High-pass filter.
         * 
         * Mathematical condition:
         *   \sum_{k} b_{k} = 0
         */
        R += sq(sum(b) - 0);


        /**
         * (C5): Orthogonality of scaling and wavelet functions.
         *
         * Mathematical condition:
         *   \sum_{k} a_{k} b_{k + 2m} = 0 \quad \forall m \in \mathbb{Z}
         */
        /* This should be automatically satisfied (confirm).
        arma::Mat<double> M5 (N+1, 3*N, arma::fill::zeros);
        for (unsigned i = 0; i < N+1; i++) {
            M5.submat(arma::span(i,i), arma::span(N,2*N-1)) = b.t();
            M5.row(i) = rowshift(M5.row(i), 2 * (N/2 - i));
        }
        M5 = M5.submat(arma::span::all, arma::span(N,2*N-1));
        R += sum(square(M5 * a));
        */
        
    }
    
    return (m_lambda/2.) * R;
}

arma::Col<double> Wavenet::SparseTermDeriv (const arma::Col<double>& y) {
    return GiniCoeffDeriv(y);
}

arma::Mat<double> Wavenet::SparseTermDeriv (const arma::Mat<double>& Y) {
    return GiniCoeffDeriv(Y);
}

arma::Col<double> Wavenet::RegTermDeriv    (const arma::Col<double>& a) {
    
    // Taking the derivative of each term in the sum of squared
    // deviations from the Kroeneker deltea, in the regularion term
    // of the cost function.
    //  - First term is the outer derivative.
    //  - Second term is the inner derivative.
    
    const int N = a.n_elem;
    arma::Col<double> D (N, arma::fill::zeros); // Derivative

    // -- Get outer derivative
    arma::Mat<double> M (N+1, 3*N, arma::fill::zeros);
    
    for (unsigned i = 0; i < N+1; i++) {
        M.submat(arma::span(i,i), arma::span(N,2*N-1)) = a.t();
        M.row(i) = rowshift(M.row(i), 2 * (N/2 - i));
    }
    M = M.submat(arma::span::all, arma::span(N,2*N-1));
    
    arma::Col<double> kroenecker_delta (N+1, arma::fill::zeros);
    kroenecker_delta(N/2) = 1;
    
    arma::Col<double> outer = 2 * (M * a - kroenecker_delta);
    
    // -- Get inner derivative.
    arma::Mat<double> inner (N + 1, N, arma::fill::zeros);
    
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
    
    D += (outer.t() * inner).t();
    
    // Wavelet part.
    if (m_wavelet) {

        arma::Col<double> b     (N, arma::fill::zeros);
        arma::Col<double> bsign (N, arma::fill::zeros);
        for (unsigned i = 0; i < N; i++) {
            b    (i) = pow(-1, i) * a(N - i - 1);
            bsign(i) = pow(-1, N - i - 1);
        }
        arma::Col<double> kroenecker_delta (N+1, arma::fill::zeros);
        kroenecker_delta(N/2) = 1;
        
        // C1
        D += 2 * (sum(a) - sqrt(2)) * arma::Col<double> (N, arma::fill::ones);
        
        // C3
        // -- Get outer derivative
        arma::Mat<double> M3 (N+1, 3*N, arma::fill::zeros);
        for (unsigned i = 0; i < N+1; i++) {
            M3.submat(arma::span(i,i), arma::span(N,2*N-1)) = b.t();
            M3.row(i) = rowshift(M3.row(i), 2 * (N/2 - i));
        }
        M3 = M3.submat(arma::span::all, arma::span(N,2*N-1));
        arma::Col<double> outer3 = 2 * (M3 * b - kroenecker_delta);
        
        // -- Get inner derivative.
        arma::Mat<double> inner3 (N + 1, N, arma::fill::zeros);
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
        D += (outer3.t() * inner3).t();
        

        // C4
        D += 2 * (sum(b) - 0) * bsign;
        
        // C5
        /* This should be automatically satisfied (confirm).
        // -- Get outer derivative
        arma::Mat<double> M5 (N+1, 3*N, arma::fill::zeros);
        for (unsigned i = 0; i < N+1; i++) {
            M5.submat(arma::span(i,i), arma::span(N,2*N-1)) = b.t();
            M5.row(i) = rowshift(M5.row(i), 2 * (N/2 - i));
        }
        M5 = M5.submat(arma::span::all, arma::span(N,2*N-1));
        arma::Col<double> outer5 = 2 * (M * b - kroenecker_delta);
        
        // -- Get inner derivative.
        arma::Mat<double> inner5 (N + 1, N, arma::fill::zeros);
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
    }
    
    return (m_lambda/2.) * D;
}

double Wavenet::cost (const arma::Col<double>& y) {

    // Sparsity term: Gini coefficient.
    double S = SparseTerm(y);
    
    // Regularisation term.
    double R = RegTerm(filter());
    
    // Sum.
    double J = S + R;
    
    return J;
}

double Wavenet::cost (const arma::Mat<double>& Y) {
    arma::Col<double> y = vectorise(Y);
    return cost(y);
}

arma::field< arma::Mat<double> > Wavenet::costMap (const arma::Mat<double>& X, const double& range, const unsigned& Ndiv) {
    return costMap(std::vector< arma::Mat<double> > ({X}), range, Ndiv);
}

arma::field< arma::Mat<double> > Wavenet::costMap (const std::vector< arma::Mat<double> >& X, const double& range, const unsigned& Ndiv) {
    
    arma::field< arma::Mat<double> > costs (3,1); // { J, S, R }
    costs.for_each( [Ndiv](arma::Mat<double>& m) { m.zeros(Ndiv, Ndiv); } );

    const unsigned nExample = X.size();
    
    INFO("  Start traversing filter grid.");
    for (unsigned i = 0; i < Ndiv; i++) {
        INFO("    Doing %d out of %d.", i, Ndiv);
        for (unsigned j = 0; j < Ndiv; j++) {
            VERBOSE("      Doing %d/%d out of %d.", i, j, Ndiv);
            double a1 = (2*j/double(Ndiv - 1) - 1) * range;
            double a2 = (2*i/double(Ndiv - 1) - 1) * range;
            setFilter({a1, a2});
            for (unsigned iExample = 0; iExample < nExample; iExample++) {
                std::vector< std::vector< arma::field< arma::Col<double> > > > Activations = forward(X.at(iExample));
                arma::Mat<double> Y = coeffsFromActivations(Activations);
                costs(0,0).submat(arma::span(i,i),arma::span(j,j)) += cost(Y);
                costs(1,0).submat(arma::span(i,i),arma::span(j,j)) += SparseTerm(Y);
                costs(2,0).submat(arma::span(i,i),arma::span(j,j)) += RegTerm(m_filter);
            }
        }
    }
    costs(0,0) /= (double) nExample;
    costs(1,0) /= (double) nExample;
    costs(2,0) /= (double) nExample;
    
    // ...
    
    return costs;
}


// High-level learnings method(s).
// -------------------------------------------------------------------

arma::Mat<double> Wavenet::basisFunction1D (const unsigned& N, const unsigned& i) {
    if (!isRadix2(N)) {
        WARNING("Cannot produce 1D basis function with length %d. Exiting.", N);
        return arma::Mat<double>();
    }
    if (i >= N) {
        WARNING("Requested index (%d) is out of bounds with length %d. Exiting.", i, N);
        return arma::Mat<double>();
    }
    const unsigned m = log2(N);
    if (!m_hasCachedOperators || size(m_cachedLowpassOperators, 0) < m) { cacheOperators(m - 1); }
    arma::Mat<double> Y (N, 1, arma::fill::zeros);
    Y(i, 0) = 1.;
    return inverse(Y);
}

arma::Mat<double> Wavenet::basisFunction2D (const unsigned& nRows, const unsigned& nCols, const unsigned& irow, const unsigned& icol) {
    if (!isRadix2(nRows) || !isRadix2(nCols)) {
        WARNING("Cannot produce 2D basis function for shape {%d, %d}. Exiting.", nRows, nCols);
        return arma::Mat<double>();
    }
    if (irow >= nRows || icol >= nCols) {
        WARNING("Requested indices (%d, %d) are out of bounds with shape {%d, %d}. Exiting.", irow, icol, nRows, nCols);
        return arma::Mat<double>();
    }
    const unsigned n = log2(nRows);
    const unsigned m = log2(nCols);
    if (!m_hasCachedOperators || size(m_cachedLowpassOperators, 0) < std::max(m,n)) { cacheOperators(std::max(m,n) - 1); }
    arma::Mat<double> Y (nRows, nCols, arma::fill::zeros);
    Y(irow, icol) = 1.;
    return inverse(Y);
}

arma::Mat<double> Wavenet::basisFunction (const unsigned& nRows, const unsigned& nCols, const unsigned& irow, const unsigned& icol) {
    if (nCols == 1) {
        assert(icol < 1);
        return basisFunction1D(nRows, irow);
    } else if (nRows == 1) {
        assert(irow < 1);
        return basisFunction1D(nCols, icol);
    } else {
        return basisFunction2D(nRows, nCols, irow, icol);
    }
}



 // Low-level learnings method(s).
// -------------------------------------------------------------------

void Wavenet::addMomentum   (const arma::Col<double>& momentum) {
    if (m_momentum.n_elem > 0) {
        m_momentum += momentum;
    } else {
        m_momentum  = momentum;
    }
    return;
}

void Wavenet::scaleMomentum (const double& factor) {
    m_momentum = m_momentum * factor;
    return;
}

void Wavenet::clear () {
    scaleMomentum(0.);
    clearFilterLog();
    clearCostLog();
    clearCachedOperators();
    return;
}

void Wavenet::update (const arma::Col<double>& gradient) {
    
    /**
     * @TODO: Add numerical guard against diverging solution.
     **/
    
    // Compute effective inertia, if necessary, depending on set inertia time scale.
    const unsigned steps = m_costLog.size();
    double effectiveInertita = (m_inertiaTimeScale > 0. ? m_inertia * m_inertiaTimeScale / (m_inertiaTimeScale + float(steps)): m_inertia);
    
    // Update.
    scaleMomentum( effectiveInertita ); /* scaleMomentum( m_inertia ); */
    addMomentum( - m_alpha * gradient);
    setFilter( m_filter + m_momentum );
    return;
}

void Wavenet::cacheOperators (const unsigned& m) {
    
    clearCachedOperators();

    m_cachedLowpassOperators .set_size(m + 1, 1);
    m_cachedHighpassOperators.set_size(m + 1, 1);

    for (unsigned i = 0; i <= m; i++) {
        m_cachedLowpassOperators (i, 0) = LowpassOperator (m_filter, i);
        m_cachedHighpassOperators(i, 0) = HighpassOperator(m_filter, i);
    }

    m_hasCachedOperators = true;
    return;
}

void Wavenet::clearCachedOperators () {
    m_cachedLowpassOperators .reset();
    m_cachedHighpassOperators.reset();
    m_hasCachedOperators = false;
    return;
}

void Wavenet::cacheWeights (const unsigned& m) {
    
    INFO("Caching arma::Matrix weights (%d).", m);
    
    clearCachedWeights();
    if (!m_hasCachedOperators) { cacheOperators(m); }
    
    const unsigned N = m_filter.n_elem;
    
    m_cachedLowpassWeights .set_size(m + 1, N);
    m_cachedHighpassWeights.set_size(m + 1, N);
    
    for (unsigned i = 0; i <= m; i++) {
    
        arma::Col<double> t (N, arma::fill::zeros);
        
        for (unsigned k = 0; k < N; k++) {
            t.fill(0);
            t(k) = 1;
            m_cachedHighpassWeights(i, k) = HighpassOperator(t, i);
            m_cachedLowpassWeights (i, k) = LowpassOperator (t, i);
        }
        
    }
    
    m_hasCachedWeights = true;
    return;
}

void Wavenet::clearCachedWeights () {
    m_cachedLowpassWeights .reset();
    m_cachedHighpassWeights.reset();
    m_hasCachedWeights = false;
    return;
}

arma::Col<double> Wavenet::lowpassfilter      (const arma::Col<double>& x) {
    const unsigned m = log2(x.n_elem);
    if (!m_hasCachedOperators || size(m_cachedLowpassOperators, 0) < m) { cacheOperators(m); }
    assert(m > 0);
    return m_cachedLowpassOperators(m - 1, 0) * x;
}

arma::Col<double> Wavenet::highpassfilter     (const arma::Col<double>& x) {
    const unsigned m = log2(x.n_elem);
    if (!m_hasCachedOperators || size(m_cachedHighpassOperators, 0) < m) { cacheOperators(m); }
    assert(m > 0);
    return m_cachedHighpassOperators(m - 1, 0) * x;
}

arma::Col<double> Wavenet::inv_lowpassfilter  (const arma::Col<double>& y) {
    const unsigned m = log2(y.n_elem);
    if (!m_hasCachedOperators || size(m_cachedLowpassOperators, 0) <= m) { cacheOperators(m); }
    return m_cachedLowpassOperators(m, 0).t() * y;
}

arma::Col<double> Wavenet::inv_highpassfilter (const arma::Col<double>& y) {
    const unsigned m = log2(y.n_elem);
    if (!m_hasCachedOperators || size(m_cachedHighpassOperators, 0) <= m) { cacheOperators(m); }
    return m_cachedHighpassOperators(m, 0).t() * y;
}

const arma::Mat<double>& Wavenet::lowpassweight (const unsigned& level, const unsigned& filt) {
    if (!m_hasCachedWeights || size(m_cachedLowpassWeights, 0) <= level) { cacheWeights(level); }
    return m_cachedLowpassWeights(level, filt);
}

const arma::Mat<double>& Wavenet::highpassweight (const unsigned& level, const unsigned& filt) {
    if (!m_hasCachedWeights || size(m_cachedHighpassWeights, 0) <= level) { cacheWeights(level); }
    return m_cachedHighpassWeights(level, filt);
}

arma::field< arma::Col<double> > Wavenet::backpropagate (const arma::Col<double>& delta, arma::field< arma::Col<double> > activations) {
    
    const unsigned N = m_filter.n_elem;
    const unsigned m = size(activations, 0) - 1;
    
    if (!m_hasCachedWeights) { cacheWeights(m); }
    
    arma::Col<double> Delta  (N, arma::fill::zeros);
    
    arma::Col<double> delta_LP (1), delta_HP (1), delta_new_LP, delta_new_HP, activ_LP;
    delta_LP.row(0) = delta.row(0);
    if (size(delta,0) > 1) {
        // Guard for 1D cases.
        delta_HP.row(0) = delta.row(1);
    }
    

    arma::Mat<double> error_LP, error_HP;
    for (unsigned i = 0; i < m; i++) {
        
        activ_LP = activations(i + 1, 0);

        // * Lowpass
        error_LP = outerProduct(delta_LP, activ_LP);
       
        for (unsigned k = 0; k < N; k++) {
            Delta (k) += frobeniusProduct( lowpassweight (i, k), error_LP );
        }
        
        // * Highpass
        error_HP = outerProduct(delta_HP, activ_LP);
      
        for (unsigned k = 0; k < N; k++) {
            Delta (k) += frobeniusProduct( highpassweight (i, k), error_HP );
        }

        // * Next level.
        delta_LP = inv_lowpassfilter(delta_LP) + inv_highpassfilter(delta_HP);
        if (i != m - 1) {
            delta_HP = delta(arma::span( pow(2, i + 1), pow(2, i + 2) - 1 ));
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

void Wavenet::batchTrain (arma::Mat<double> X) {
    
    /// Main method for training wavenet instance. 
    /// 
    /// Performs (1) forward propagation to find activation of all nodes and (2) back-progation to find error terms of all matrix weights (i.e. wavelet filter coefficients).

    const unsigned nRows = size(X, 0);
    const unsigned nCols = size(X, 1);

    // Perform forward transform of input 'X', and get activations of all nodes in wavenet.
    std::vector< std::vector< arma::field< arma::Col<double> > > > Activations = forward(X);
    
    // Given the complete set of node activations, get the (nRows x nCols) set of wavelet coefficients.
    arma::Mat< double > Y (size(X));
    for (unsigned icol = 0; icol < nCols; icol++) {
        Y.col(icol) = coeffsFromActivations( Activations.at(1).at(icol) );
    }
    
    // Compute sparsity error on wavelet (NB: not filter) coefficents.
    arma::Mat< double > delta = SparseTermDeriv(Y), new_delta(size(delta)); 
    
    // Back-propagate these errors to errors on the transfer matrix weights (i.e. filter coefficients).
    arma::Col<double> Delta  (size(m_filter), arma::fill::zeros);
    arma::Col<double> weight (size(m_filter), arma::fill::zeros);

    // -- Columns.
    for (unsigned icol = 0; icol < nCols; icol++) {
        arma::field< arma::Col<double> > outField = backpropagate(delta.col(icol), Activations.at(1).at(icol));
        Delta += outField(0,0);
        new_delta.col(icol) = outField(1,0);
    }

    // -- Rows.
    for (unsigned irow = 0; irow < nRows; irow++) {
        arma::field< arma::Col<double> > outField = backpropagate(new_delta.row(irow).t(), Activations.at(0).at(irow));
        Delta += outField(0,0);
    }
    
    // Compute sparsity error on filter coefficients.
    arma::Col<double> regularisation = RegTermDeriv(m_filter);
    
    // Compute combined errror
    Delta += regularisation;

    // Add current combined (back-propagated sparsity and regularisation) errors to batch queue.
    m_batchQueue.push_back(Delta);
    m_costLog.back() += cost(Y);

    // If batch queue has reach maximal batch size, flush the queue.
    if (m_batchQueue.size() >= m_batchSize) { flushBatchQueue(); }
    
    return;
}


void Wavenet::flushBatchQueue () {
    // If batch is empty, exit.
    if (!m_batchQueue.size()) { return; }

    // Compute batch-averaged gradient.
    arma::Col<double> gradient (size(m_filter), arma::fill::zeros);
    for (unsigned i = 0; i < m_batchQueue.size(); i++) {
        gradient += m_batchQueue.at(i);
    }
    gradient /= float(m_batchQueue.size());
    
    // Update with batch-averaged gradient.
    this->update(gradient);

    // Update cost log.
    m_costLog.back() /= float(m_batchQueue.size());
    m_costLog.push_back(0);
    
    // Clear batch queue.
    m_batchQueue.clear();
    
    return;
}


 // Miscellaneous.
// -------------------------------------------------------------------
arma::Col<double> Wavenet::coeffsFromActivations (const arma::field< arma::Col<double> >& activations) {
    
    const unsigned m = size(activations, 0) - 1;
    
    arma::Col<double> y (pow(2, m), arma::fill::zeros);

    y(arma::span(0,0)) = activations(0, 0);
    for (unsigned i = 0; i < m; i++) {
        y( arma::span(pow(2, i), pow(2, i + 1) - 1) ) = activations(i, 1);
    }
    
    return y;
    
}

arma::Mat<double> Wavenet::coeffsFromActivations (const std::vector< std::vector< arma::field< arma::Col<double> > > >& Activations) {
    
    const unsigned nRows = Activations.at(0).size();
    const unsigned nCols = Activations.at(1).size();
    
    arma::Mat<double> Y (nRows, nCols, arma::fill::zeros);
    
    for (unsigned icol = 0; icol < nCols; icol++) {
        Y.col(icol) = coeffsFromActivations(Activations.at(1).at(icol));
    }
    
    return Y;
    
}

} // namespace
