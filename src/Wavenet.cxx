#include "Wavenet/Wavenet.h"

namespace wavenet {

/// Set method(s).
// -----------------------------------------------------------------------------

bool Wavenet::setFilter (const arma::Col<double>& filter) {

    // Perform checks.
    if (filter.is_empty()) {
        WARNING("Input filter is empty.");
        return false;
    }

    if (filter.n_rows % 2) {
        WARNING("Input filter length is not a multiple of 2.");
        return false;
    }

    // Set wavenet filter coeffients.
    m_filter = filter;

    // Add to filter coefficent log.
    m_filterLog.push_back(m_filter);

    // Clear operators cached with the previous filter.
    clearCachedOperators_();
    
    // If the filter size is changes, resize the momentum vector accordingly.
    if (m_momentum.n_elem != m_filter.n_elem) {
        m_momentum.zeros(size(m_filter));
    }
    
    return true;
}


/// Get method(s).
// -----------------------------------------------------------------------------

double Wavenet::firstCost () const  { 

    // Can only get first cost in log, if log is non-empty.
    if (m_costLog.size() > 0) {
        return m_costLog[0];
    }

    return -1.;
}

double Wavenet::lastCost () const  { 

    // Can only get last cost in log, if log is non-empty.
    if (m_costLog.size() > 0) {

        // If the last entry is non-zero, use this.
        if (m_costLog.back() > 0) {
            
            if (m_batchQueue.size() > 0) {

                // If the batch queue is non-empty, scale the last cost by 
                // number of entries in the batch queue.
                return m_costLog.back() / float(m_batchQueue.size());

            } else {

                // Otherwise return the last entry.
                return m_costLog.back();   
            }

        } else if (m_costLog.size() > 1) {

            // Otherwise, and if possible, get next-to-last cost in log, which
            // will be properly normalised.
            return m_costLog[m_costLog.size() - 2];

        }
    }

    return -1.;
}


/// Print method(s).
// -----------------------------------------------------------------------------

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


/// Storage method(s).
// -----------------------------------------------------------------------------

void Wavenet::save (Snapshot snap) const {

    DEBUG("Saving snapshot '%s'.", snap.file().c_str());

    // Perform checks.
    if (strcmp(snap.file().substr(0,1).c_str(), "/") == 0) {
        WARNING("File '%s' not accepted. Only accepting realtive paths.", snap.file().c_str());
        return;
    }
    
    if (snap.exists()) {
        DEBUG("File '%s' already exists. Overwriting.", snap.file().c_str()); 
    }
    
    if (snap.file().find("/") != std::string::npos) {
        std::string dir = snap.file().substr(0,snap.file().find_last_of("/"));
        if (!dirExists(dir)) {
            WARNING("Directory '%s' does not exist. Creating it.", dir.c_str());
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


/// High-level learning method(s).
// -----------------------------------------------------------------------------

bool Wavenet::train (arma::Mat<double> X) {
    
    // Impose guard against exections, chiefly from NaN due to diverging
    // solutions.
    try {

        // Initialise size variable(s).
        const unsigned nRows = size(X, 0); // Number of rows.
        const unsigned nCols = size(X, 1); // Number of columns.

        // Perform forward transform of input X and get activations of all nodes in 
        // wavenet.
        Activations2D_t Activations = forward_(X);

        // Given the complete set of node activations, get the corresponding 
        // (nRows x nCols) set of wavelet coefficients.
        arma::Mat<double> Y (size(X)); // Matrix of wavelet coefficients.
        for (unsigned icol = 0; icol < nCols; icol++) {
            Y.col(icol) = coeffsFromActivations( Activations.at(1).at(icol) );
        }

        // Compute the gradient of the sparsity error on the wavelet (NB: not 
        // filter) coefficents corresponding to the input X.
        arma::Mat<double> delta = SparseTermDeriv(Y);
        
        // Given these errors, and the activations from forward transforming the 
        // input X, perform the complete, 2D backpropagation to get the resulting
        // error gradient for the filter coefficients.
        arma::Col<double> gradientSparsity = backpropagate_(delta, Activations);
        
        // Compute the gradient of the regularisation error on filter coefficients 
        // of the wavenet object.
        arma::Col<double> gradientRegularisation = lambda() * RegTermDeriv(m_filter, m_wavelet);
        
        // Compute the combined error on the filter coefficients.
        arma::Col<double> gradientCombined = gradientSparsity + gradientRegularisation;

        // Add current combined (back-propagated sparsity and regularisation) 
        // gradient to the batch queue.
        m_batchQueue.push_back(gradientCombined);

        // Add the combined (sparsity and regularisation) cost of the wavelet 
        // coefficients Y to the latest entry in the cost log.
        m_costLog.back() += cost(Y);

        // If batch queue has reached batch size, flush the queue.
        if (m_batchQueue.size() >= m_batchSize) { flushBatchQueue_(); }
        
        return true;

    } catch (const std::exception& e) {

        // If an error occured, print it.
        ERROR("%s", e.what());
        ERROR("Most likely due to diverging solution. Try lowering the learning rate (alpha) or the regularisation term (lambda). Exiting.");

    }
    return false;
}

void Wavenet::clear () {
    scaleMomentum_(0.);
    clearFilterLog();
    clearCostLog();
    clearCachedOperators_();
    return;
}


/// High-level cost method(s).
// -----------------------------------------------------------------------------

double Wavenet::cost (const arma::Col<double>& y) {

    // Sparsity term: Gini coefficient.
    double S = SparseTerm(y);
    
    // Regularisation term.
    double R = lambda() * RegTerm(m_filter, m_wavelet);
    
    // Sum.
    double J = S + R;
    
    return J;
}

double Wavenet::cost (const arma::Mat<double>& Y) {

    // Turn matrix into vector, and call Wavenet::cost(arma::Col<double>).
    arma::Col<double> y = vectorise(Y);

    return cost(y);
}

std::vector< arma::Mat<double> > Wavenet::costMap (const std::vector< arma::Mat<double> >& X, const double& range, const unsigned& Ndiv) {

    // Initialise the Armadillo field of output cost map matrices.
    std::vector< arma::Mat<double> > costs (3); // { Combined, Sparsity, Regularisation }
    for (arma::Mat<double>& M : costs) {
        M.zeros(Ndiv, Ndiv);
    }
    // Initialise number of training examples.
    const unsigned nExample = X.size();
    
    INFO("  Start traversing filter grid.");
    for (unsigned i = 0; i < Ndiv; i++) {
        INFO("    Doing %d out of %d.", i, Ndiv);
        for (unsigned j = 0; j < Ndiv; j++) {
            VERBOSE("      Doing %d/%d out of %d.", i, j, Ndiv);

            // Get the values for the filter coefficients, given range to scan, 
            // the number of divisions, and the current entry.
            double a1 = (2*j/double(Ndiv - 1) - 1) * range;
            double a2 = (2*i/double(Ndiv - 1) - 1) * range;
            setFilter({a1, a2});

            // Loop training examples.
            for (unsigned iExample = 0; iExample < nExample; iExample++) {

                // Perform forward transform.
                std::vector< std::vector< arma::field< arma::Col<double> > > > Activations = forward_(X.at(iExample));

                // Get wavelet coefficients.
                arma::Mat<double> Y = coeffsFromActivations(Activations);

                // Compute costs.
                costs.at(0).submat(arma::span(i,i),arma::span(j,j)) += cost(Y);
                costs.at(1).submat(arma::span(i,i),arma::span(j,j)) += SparseTerm(Y);
                costs.at(2).submat(arma::span(i,i),arma::span(j,j)) += lambda() * RegTerm(m_filter, m_wavelet);
            }
        }
    }

    // Normalise costs by the number of training examples.
    costs.at(0) /= (double) nExample;
    costs.at(1) /= (double) nExample;
    costs.at(2) /= (double) nExample;
    
    return costs;
}


/// Basis function method(s).
// -----------------------------------------------------------------------------

arma::Mat<double> Wavenet::basisFunction1D (const unsigned& nRows, const unsigned& irow) {

    // Perform check(s).
    if (!isRadix2(nRows)) {
        WARNING("Cannot produce 1D basis function with length %d. Exiting.", nRows);
        return arma::Mat<double>();
    }
    if (irow >= nRows) {
        WARNING("Requested index (%d) is out of bounds with length %d. Exiting.", irow, nRows);
        return arma::Mat<double>();
    }

    // Initialise size variable(s).
    const unsigned m = log2(nRows);

    // Ensure matrix operators are cached.
    if (!m_hasCachedOperators || size(m_cachedLowpassOperators, 0) < m) { cacheOperators_(m - 1); }

    // Perform inverse transform.
    arma::Mat<double> Y (nRows, 1, arma::fill::zeros);
    Y(irow, 0) = 1.;
    return inverse_(Y);
}

arma::Mat<double> Wavenet::basisFunction2D (const unsigned& nRows, const unsigned& nCols, const unsigned& irow, const unsigned& icol) {
    
    // Perform check(s).
    if (!isRadix2(nRows) || !isRadix2(nCols)) {
        WARNING("Cannot produce 2D basis function for shape {%d, %d}. Exiting.", nRows, nCols);
        return arma::Mat<double>();
    }
    if (irow >= nRows || icol >= nCols) {
        WARNING("Requested indices (%d, %d) are out of bounds with shape {%d, %d}. Exiting.", irow, icol, nRows, nCols);
        return arma::Mat<double>();
    }

    // Initialise size variable(s).
    const unsigned m = log2(nRows);
    const unsigned n = log2(nCols);

    // Ensure matrix operators are cached.
    if (!m_hasCachedOperators || size(m_cachedLowpassOperators, 0) < std::max(m,n)) { cacheOperators_(std::max(m,n) - 1); }

    // Perform inverse transform.
    arma::Mat<double> Y (nRows, nCols, arma::fill::zeros);
    Y(irow, icol) = 1.;
    return inverse_(Y);
}

arma::Mat<double> Wavenet::basisFunction (const unsigned& nRows, const unsigned& nCols, const unsigned& irow, const unsigned& icol) {

    // Determine dimension.
    if (nCols == 1) {

        // Row vector.
        assert(icol < 1);
        return basisFunction1D(nRows, irow);

    } else if (nRows == 1) {

        // Column vector.
        assert(irow < 1);
        return basisFunction1D(nCols, icol);

    } else {

        // Matrix.
        return basisFunction2D(nRows, nCols, irow, icol);
    }
}


/// 1D wavenet transform method(s).
// -----------------------------------------------------------------------------

Activations1D_t Wavenet::forward_ (const arma::Col<double>& x) {

    // Initialise size variable(s).
    const unsigned m = log2(x.n_elem); // Number of wavenet layers.
    
    // Initialise output field of wavenet node activations.
    Activations1D_t activations(m + 1, 2);
    
    /**
     * Structure of the 1D activation type: 
     *   (m, 0) = Low-pass  coeffs. at level m
     *   (m, 1) = High-pass coeffs. at level m
     */
    
    // Make a mutable copy of the input, position space vector, to be forward 
    // transformed.
    arma::Col<double> x_current = x;

    // Loop wavenet layers.
    for (unsigned i = m; i --> 0; ) {

        // Store low-pass filter activations.
        activations(i, 0) = lowpassfilter_ (x_current);

        // Store high-pass filter activations.
        activations(i, 1) = highpassfilter_(x_current);

        // Update vector as the low-pass filtered version, and proceed to the 
        // next level.
        x_current = activations(i, 0);
    }

    // Add highest-level low-pass activations: the original position space
    // vector.
    activations(m, 0) = x;

    // Add highest-level high-pass activations: all zeros, size no details are 
    // missing at this level.
    activations(m, 1) = zeros(size(x));
    
    return activations;
}

arma::Col<double> Wavenet::inverse_ (const arma::Col<double>& y) {

    // Initialise size variable(s).
    const unsigned m = log2(y.n_elem); // Number of wavenet layers.
    
    // Initialise output vector (position space) to size 1.
    arma::Col<double> x (1, arma::fill::ones);

    // Set the value to the lowest-scale wavelet coefficient (the "average"
    // coefficient).
    x.fill(y(0));

    // Loop wavenet layers.
    for (unsigned i = 0; i < m; i++) {

        // Perform the inverse low-pass operation of the position space vector.
        x  = inv_lowpassfilter_ (x);

        // Perform the inverse high-pass operation on the appropriate wavelet 
        // coefficints, and add to the position space vector, to recover the
        // missing details at this level.
        x += inv_highpassfilter_(y( arma::span(pow(2, i), pow(2, i + 1) - 1) ));
    }
    
    return x;
}

std::vector< arma::Col<double> > Wavenet::backpropagate_ (const arma::Col<double>& delta, Activations1D_t activations) {

    // Initialise size variable(s).
    const unsigned N = m_filter.n_elem; // Number of filter coefficients.
    const unsigned m = size(activations, 0) - 1; // Number of wavenet layers.
    
    // Cache filter coefficient weight matrices, if necessary.
    if (!m_hasCachedWeights) { cacheWeights_(m); }

    // Initialise the gradient (or filter coefficient error) vector.
    arma::Col<double> gradient (N, arma::fill::zeros);
    
    // Initialise the various low- and high-pass error vectors.
    arma::Col<double> delta_LP (1); // Error on current low-pass nodes.
    arma::Col<double> delta_HP (1); // Error on current high-pass nodes.
    arma::Col<double> delta_new_LP; // Error on next layer's low-pass nodes.
    arma::Col<double> delta_new_HP; // Error on next layer's high-pass nodes.
    arma::Col<double> activ_LP;     // Activations of current low-pass nodes.

    // Initialise the error on the last low-pass node as the error on the 
    // lowest-scale wavelet coefficient.
    delta_LP.row(0) = delta.row(0);

    // Initialise the error on the last high-pass node as the error on the 
    // second-lowest-scale wavelet coefficient, if applicable.
    if (size(delta,0) > 1) {
        // Guard for 1D cases.
        delta_HP.row(0) = delta.row(1);
    }
    
    // Initialise error matrices for low- and high-pass matrix operators.
    arma::Mat<double> error_LP;
    arma::Mat<double> error_HP;

    // Iterate wavelet scales/neural network layers backwards.
    for (unsigned i = 0; i < m; i++) {
        
        // Get activations at current layer.
        activ_LP = activations(i + 1, 0);

        // Compute error on low-pass matrix operator.
        error_LP = outerProduct(delta_LP, activ_LP);
       
        // Add errors to filter coefficients.
        for (unsigned k = 0; k < N; k++) {
            gradient (k) += frobeniusProduct( lowpassweight_ (i, k), error_LP );
        }
        
        // Compute error on high-pass matrix operator.
        error_HP = outerProduct(delta_HP, activ_LP);
      
        // Add errors to filter coefficients.
        for (unsigned k = 0; k < N; k++) {
            gradient (k) += frobeniusProduct( highpassweight_ (i, k), error_HP );
        }

        // Go to next level, by performing a two single-layer inverse filter 
        // operations, and combining the result (i.e. performing one step in the 
        // inverse wavelet transform)
        delta_LP = inv_lowpassfilter_(delta_LP) + inv_highpassfilter_(delta_HP);
        if (i != m - 1) {
            // If at intermediate level, get high-pass errors from the 
            // corresponding wavelet coefficient errors.
            delta_HP = delta(arma::span( pow(2, i + 1), pow(2, i + 2) - 1 ));
        } else {
            // Otherwise, if at last level, set the high-pass errors to be zero; 
            // they're not used anyway.
            delta_HP.copy_size(delta_LP);
            delta_HP.zeros();
        }
        
    }

    // Initialise output vector.    
    std::vector< arma::Col<double> > output(2);

    // Store (1) erorrs on filter coefficients and (2) "errors on input", for 
    // use in 2D backprogation.
    output.at(0) = gradient;
    output.at(1) = delta_LP;

    return output;
}


/// 2D wavenet transform method(s).
// -----------------------------------------------------------------------------

Activations2D_t Wavenet::forward_ (const arma::Mat<double>& X) {
    
    // Initialise size variable(s).
    const unsigned nRows = size(X,0); // Number of rows.
    const unsigned nCols = size(X,1); // Number of columns.
    
    // Initialise output field of wavenet node activations.
    Activations2D_t Activations(2);
    Activations.at(0).resize(nRows);
    Activations.at(1).resize(nCols);
    
    /**
     * Structure of the 2D activation type: 
     * (Note: transform is performed in a row-major fashion.)
     *   [0][irow](m, 0) = Low-pass  coeffs. at level m, for row irow
     *   [0][irow](m, 1) = High-pass coeffs. at level m, for row irow
     *   [1][icol](m, 0) = Low-pass  coeffs. at level m, for col icol
     *   [1][icol](m, 1) = High-pass coeffs. at level m, for col icol
     */

    // Make a mutable copy of the input, position space matrix, to be forward 
    // transformed.
    arma::Mat<double> Y = X;

    // Forward transform rows.
    for (unsigned irow = 0; irow < nRows; irow++) {

        // Make mutable copy of the irow'th row.
        arma::Col<double> x = Y.row(irow).t();

        // Forward transform the row to get all node activations.
        arma::field< arma::Col<double> > activations = forward_(x);

        // Extract wavelet coefficients and store in irow'th row of matrix being
        // transformed.
        Y.row(irow) = coeffsFromActivations(activations).t();

        // Store activations in output field.
        Activations.at(0).at(irow) = std::move(activations);
    }
    
    // Forward transform resulting columns.
    for (unsigned icol = 0; icol < nCols; icol++) {

        // Make mutable copy of the icol'th column of the partially transformed 
        // matrix.
        arma::Col<double> x = Y.col(icol);

        // Forward transform the column to get all node activations.
        arma::field< arma::Col<double> > activations = forward_(x);

        // Extract wavelet coefficients and store in icol'th column of matrix 
        // being transformed.
        Y.col(icol) = coeffsFromActivations(activations);

        // Store activations in output field.
        Activations.at(1).at(icol) = std::move(activations);
    }
    
    return Activations;
}

arma::Mat<double> Wavenet::inverse_ (const arma::Mat<double>& Y) {
    
    // Initialise size variable(s).
    const unsigned nRows = size(Y, 0); // Number of rows.
    const unsigned nCols = size(Y, 1); // Number of columns.
    
    /**
     * (Note: Since the forward transform is performed in a row-major fashion, 
     *  the inverse transform must be performed column-major.)
     */

    // Make a mutable copy of the input, wavelet coefficient matrix, to be 
    // inverse transformed.
    arma::Mat<double> X = Y;

    // Inverse transform columns.
    for (unsigned icol = 0; icol < nCols; icol++) {

        // Make mutable copy of the icol'th column.
        arma::Col<double> x = X.col(icol);

        // Inverse transform the column, and store in icol'th column of matrix 
        // being transformed.
        X.col(icol) = inverse_(x);
    }

    // Inverse transform resulting rows.
    for (unsigned irow = 0; irow < nRows; irow++) {

        // Make mutable copy of the irow'th row of the partially transformed 
        // matrix.        
        arma::Col<double> x = X.row(irow).t();

        // Inverse transform the row, and store in irow'th row of matrix being
        // transformed.
        X.row(irow) = inverse_(x).t();
    }
    
    return X;
}

arma::Col<double> Wavenet::backpropagate_ (const arma::Mat<double>& Delta, Activations2D_t Activations) {

    // Initialise size variable(s).
    const unsigned nRows = Activations.at(0).size(); // Number of rows;
    const unsigned nCols = Activations.at(1).size(); // Number of columns;

    /**
     * (Note: Since the forward transform is performed in a row-major fashion, 
     *  the backpropagation must be performed column-major.)
     */

    // Make a mutable copy of the input error matrix, to be backpropagated.
    arma::Mat<double> X = Delta; 
    
    // Initialise vector of error gradients on the filter coefficients.
    arma::Col<double> gradient (size(m_filter), arma::fill::zeros);

    // Initialise object for holding the output for the 1D backprogation 
    // operations.
    std::vector< arma::Col<double> > backpropOutput;

    // Backpropagate columns.
    for (unsigned icol = 0; icol < nCols; icol++) {

        // Backpropagate the icol'th column of the input error matrix.
        backpropOutput = backpropagate_(X.col(icol), Activations.at(1).at(icol));

        // Add the errors on the filter coefficients from this operation.
        gradient += backpropOutput.at(0);

        // Store the "error on the input" as the icol'th column in the matrix 
        // being transformed
        X.col(icol) = backpropOutput.at(1);
    }

    // Backpropagate resulting rows.
    for (unsigned irow = 0; irow < nRows; irow++) {

        // Backpropagate the irow'th row of the resulting error matrix.
        backpropOutput = backpropagate_(X.row(irow).t(), Activations.at(0).at(irow));

        // Add the errors on the filter coefficients from this operation.
        gradient += backpropOutput.at(0);
    }

    return gradient;
}


/// Low-level learnings method(s).
// -----------------------------------------------------------------------------

void Wavenet::flushBatchQueue_ () {

    // If batch is empty, do nothing.
    if (!m_batchQueue.size()) { return; }

    // Compute batch-averaged gradient.
    arma::Col<double> gradient (size(m_filter), arma::fill::zeros);
    for (unsigned i = 0; i < m_batchQueue.size(); i++) {
        gradient += m_batchQueue.at(i);
    }
    gradient /= float(m_batchQueue.size());
    
    // Update with batch-averaged gradient.
    this->update_(gradient);

    // Update cost log.
    m_costLog.back() /= float(m_batchQueue.size());
    m_costLog.push_back(0);
    
    // Clear batch queue.
    m_batchQueue.clear();
    
    return;
}

void Wavenet::addMomentum_ (const arma::Col<double>& gradient) {
    if (m_momentum.n_elem > 0) { m_momentum += gradient; }
    else                       { m_momentum  = gradient; }
    return;
}

void Wavenet::scaleMomentum_ (const double& factor) {
    m_momentum = m_momentum * factor;
    return;
}

void Wavenet::update_ (const arma::Col<double>& gradient) {
    
    // Compute effective inertia, if necessary, depending on set inertia time scale.
    const unsigned steps = m_costLog.size() - 1;
    double effectiveInertita = (m_inertiaTimeScale > 0. ? m_inertia * (1. - exp( - float(steps) / m_inertiaTimeScale )) : m_inertia);
    
    // Update.
    scaleMomentum_( effectiveInertita ); 
    addMomentum_( - m_alpha * gradient);
    setFilter( m_filter + m_momentum );

    return;
}

void Wavenet::cacheOperators_ (const unsigned& m) {
    
    // Clear existing cache.
    clearCachedOperators_();

    // Resize cache vectors.
    m_cachedLowpassOperators .set_size(m + 1, 1);
    m_cachedHighpassOperators.set_size(m + 1, 1);

    // Perform caching.
    for (unsigned i = 0; i <= m; i++) {
        m_cachedLowpassOperators (i, 0) = LowpassOperator (m_filter, i);
        m_cachedHighpassOperators(i, 0) = HighpassOperator(m_filter, i);
    }

    // Switch flag.
    m_hasCachedOperators = true;

    return;
}

void Wavenet::clearCachedOperators_ () {

    // Reset cache vectors.
    m_cachedLowpassOperators .reset();
    m_cachedHighpassOperators.reset();

    // Switch flag.
    m_hasCachedOperators = false;

    return;
}

void Wavenet::cacheWeights_ (const unsigned& m) {
    
    DEBUG("Caching matrix weights (%d).", m);
    
    // Clear existing cache.
    clearCachedWeights_();
    
    // Initialise size variable(s).
    const unsigned N = m_filter.n_elem;
    
    // Resize cache vectors.
    m_cachedLowpassWeights .set_size(m + 1, N);
    m_cachedHighpassWeights.set_size(m + 1, N);

    // Initialise filter coefficient vector.
    arma::Col<double> t (N, arma::fill::zeros);        
    
    // Loop frequency scales.
    for (unsigned i = 0; i <= m; i++) {

        // Loop filter coefficients.
        for (unsigned k = 0; k < N; k++) {
            // Set all filter coefficients to zero.
            t.fill(0);

            // Switch k'th filter coefficient to one.
            t(k) = 1;

            // Construct matrix operators.
            m_cachedHighpassWeights(i, k) = HighpassOperator(t, i);
            m_cachedLowpassWeights (i, k) = LowpassOperator (t, i);
        }

    }
    
    // Switch flag.
    m_hasCachedWeights = true;

    return;
}

void Wavenet::clearCachedWeights_ () {

    // Reset cache vectors.
    m_cachedLowpassWeights .reset();
    m_cachedHighpassWeights.reset();

    // Switch flag.
    m_hasCachedWeights = false;

    return;
}

arma::Col<double> Wavenet::lowpassfilter_ (const arma::Col<double>& x) {

    // Get number of wavenet levels.
    const unsigned m = log2(x.n_elem);

    // Make sure that operators are cached at least up to level m.
    if (!m_hasCachedOperators || size(m_cachedLowpassOperators, 0) < m) { cacheOperators_(m); }

    // Apply low-pass filter using cached operator.
    return m_cachedLowpassOperators(m - 1, 0) * x;
}

arma::Col<double> Wavenet::highpassfilter_ (const arma::Col<double>& x) {
    
    // Get number of wavenet levels.
    const unsigned m = log2(x.n_elem);
    
    // Make sure that operators are cached at least up to level m.
    if (!m_hasCachedOperators || size(m_cachedHighpassOperators, 0) < m) { cacheOperators_(m); }

    // Apply high-pass filter using cached operator.
    return m_cachedHighpassOperators(m - 1, 0) * x;
}

arma::Col<double> Wavenet::inv_lowpassfilter_ (const arma::Col<double>& y) {
    
    // Get number of wavenet levels.
    const unsigned m = log2(y.n_elem);
    
    // Make sure that operators are cached at least up to level m.
    if (!m_hasCachedOperators || size(m_cachedLowpassOperators, 0) <= m) { cacheOperators_(m); }
    
    // Apply inverse low-pass filter using cached operator.
    return m_cachedLowpassOperators(m, 0).t() * y;
}

arma::Col<double> Wavenet::inv_highpassfilter_ (const arma::Col<double>& y) {
    
    // Get number of wavenet levels.
    const unsigned m = log2(y.n_elem);
    
    // Make sure that operators are cached at least up to level m.
    if (!m_hasCachedOperators || size(m_cachedHighpassOperators, 0) <= m) { cacheOperators_(m); }
    
    // Apply inverse high-pass filter using cached operator.
    return m_cachedHighpassOperators(m, 0).t() * y;
}

const arma::Mat<double>& Wavenet::lowpassweight_ (const unsigned& level, const unsigned& filt) {

    // Make sure that weight matrices are cached at least up to 'level.
    if (!m_hasCachedWeights || size(m_cachedLowpassWeights, 0) <= level) { cacheWeights_(level); }

    // Return low-pass weight matrix at 'level' for filter coefficient 'filt'.
    return m_cachedLowpassWeights(level, filt);
}

const arma::Mat<double>& Wavenet::highpassweight_ (const unsigned& level, const unsigned& filt) {
    
    // Make sure that weight matrices are cached at least up to 'level.
    if (!m_hasCachedWeights || size(m_cachedHighpassWeights, 0) <= level) { cacheWeights_(level); }

    // Return high-pass weight matrix at 'level' for filter coefficient 'filt'.
    return m_cachedHighpassWeights(level, filt);
}

} // namespace
