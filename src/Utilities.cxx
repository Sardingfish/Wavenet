#include "Wavenet/Utilities.h"
#include "Wavenet/Wavenet.h" /* To create Wavenet instance. */ 

namespace wavenet {

/// Armadillo-specific functions.
arma::Col<double> PointOnNSphere (const unsigned& N, const double& rho) {
    
    // Initialise output vector of filter coefficients.
    arma::Col<double> coords (N, arma::fill::ones);
    
    // Generate point in N-dimensional filter coefficient space according not 
    // normal distribution.
    coords = arma::randn< arma::Col<double> > (N);

    // Scale point to norm one, thereby ensuring the the ensemle corresponding 
    // to uniformly random points on the unit N-sphere
    arma::arma_rng::set_seed_random();
    coords *= (1 + arma::as_scalar(arma::randn(1)) * rho) / arma::norm(coords);
    
    return coords;
}

arma::Col<double> coeffsFromActivations (const arma::field< arma::Col<double> >& activations) {
    
    // Initialise number of wavenet layers.
    const unsigned m = size(activations, 0) - 1;
    
    // Initialise out vector of wavelet coefficients.
    arma::Col<double> y (pow(2, m), arma::fill::zeros);

    // Set (0,0) or "average" coefficients.
    y(arma::span(0,0)) = activations(0, 0);
    for (unsigned i = 0; i < m; i++) {

        // Set the coefficients corresponding to layer, or frequency scale, i.
        y( arma::span(pow(2, i), pow(2, i + 1) - 1) ) = activations(i, 1);
    }
    
    return y;
}

arma::Mat<double> coeffsFromActivations (const std::vector< std::vector< arma::field< arma::Col<double> > > >& Activations) {
    
    // Initialise size variable(s).
    const unsigned nRows = Activations.at(0).size();
    const unsigned nCols = Activations.at(1).size();
    
    // Initialise output matrix of wavelet coefficients.
    arma::Mat<double> Y (nRows, nCols, arma::fill::zeros);

    // Get wavelet coefficients for each column in collection of activations.
    for (unsigned icol = 0; icol < nCols; icol++) {
        Y.col(icol) = coeffsFromActivations(Activations.at(1).at(icol));
    }
    
    return Y;
}


/// ROOT-specific functions.
#ifdef USE_ROOT

TGraph costGraph (const std::vector< double >& costLog) {
    
    // Initialise number of entries in the cost log.
    const unsigned N = costLog.size();

    // Set graph points as cost for each entry.
    double x[N], y[N];
    for (unsigned i = 0; i < N; i++) {
        x[i] = i;
        y[i] = costLog.at(i);
        
    }
    
    // Initialise TGraph of cost log.
    TGraph graph (N, x, y);
    
    return graph;
}

std::unique_ptr<TH1> MatrixToHist2D (const arma::Mat<double>& matrix, const double& range) {
    
    // Initialise size variable(s).
    const unsigned nRows = size(matrix, 0);
    const unsigned nCols = size(matrix, 1);
    
    // Initialise unique pointer to output ROOT TH2F.
    std::unique_ptr<TH1> hist (new TH2F("hist", "", nRows, -range, range, nCols, -range, range));
    
    // Fill histogram with matrix entries.
    for (unsigned irow = 0; irow < nRows; irow++) {
        for (unsigned icol = 0; icol < nCols; icol++) {
            hist->SetBinContent(irow + 1, icol + 1, matrix(irow, icol));
        }
    }
    
    return std::move(hist);
}

std::unique_ptr<TH1> MatrixToHist1D (const arma::Mat<double>& matrix, const double& range) {
    
    // Initialise size variable(s).
    const unsigned N = size(matrix, 0);
    
    // Initialise unique pointer to output ROOT TH1F.
    std::unique_ptr<TH1> hist (new TH1F("hist", "", N, -range, range) );
    
    // Fill histogram with vector entries.
    for (unsigned i = 0; i < N; i++) {
        hist->SetBinContent(i + 1, matrix(i, 0));
    }
    
    return std::move(hist);
}

std::unique_ptr<TH1> MatrixToHist (const arma::Mat<double>& matrix, const double& range) {
    
    // Determine dimension of matrix, and call appropriate specialised function.
    if (size(matrix,1) == 1) {
        return MatrixToHist1D(matrix, range);
    } else {
        return MatrixToHist2D(matrix, range);
    }
}

arma::Mat<double> HistFillMatrix2D (const TH1* hist, arma::Mat<double>& matrix) {
    
    // Check type.
    assert(dynamic_cast<const TH2F*>(hist));

    // Initialise size variable(s).
    const unsigned nRows = hist->GetYaxis()->GetNbins();
    const unsigned nCols = hist->GetXaxis()->GetNbins();
    
    // Check that histogram and matrix sizes agree.
    assert(nRows == size(matrix,0));
    assert(nCols == size(matrix,1));

    // Reset output matrix.
    matrix.zeros();
    
    // Fill matrix entries from histogram.
    for (unsigned irow = 0; irow < nRows; irow++) {
        for (unsigned icol = 0; icol < nCols; icol++) {
            matrix(irow, icol) = hist->GetBinContent(irow + 1, icol + 1);
        }
    }
    
    return matrix;
}

arma::Mat<double> HistFillMatrix1D (const TH1* hist, arma::Mat<double>& matrix) {
    
    // Check type.
    assert(dynamic_cast<const TH1F*>(hist));

    // Initialise size variable(s).
    const unsigned N = hist->GetYaxis()->GetNbins();
    
    // Check that histogram and matrix (vector) sizes agree.
    assert(N == size(matrix,0));
    
    // Reset output matrix (vector).
    matrix.zeros();
    
    // Fill matrix (vector) entries from histogram.
    for (unsigned i = 0; i < N; i++) {
        matrix(i, 0) = hist->GetBinContent(i + 1);
    }
    
    return matrix;
}

arma::Mat<double> HistFillMatrix (const TH1* hist, arma::Mat<double>& matrix) {
   
    // Determine dimension of matrix, and call appropriate specialised function.
    if (size(matrix,1) == 1) {
        return HistFillMatrix1D(hist, matrix);
    } else {
        return HistFillMatrix2D(hist, matrix);
    }
}

arma::Mat<double> HistToMatrix2D (const TH1* hist) {
    
    // Check type.
    assert(dynamic_cast<const TH2F*>(hist));

    // Initialise size variable(s).
    const unsigned nRows = hist->GetYaxis()->GetNbins();
    const unsigned nCols = hist->GetXaxis()->GetNbins();
    
    // Initialise output matrix.
    arma::Mat<double> matrix (nRows, nCols, arma::fill::zeros);
    
    // Fill matrix entries from histogram.
    for (unsigned irow = 0; irow < nRows; irow++) {
        for (unsigned icol = 0; icol < nCols; icol++) {
            matrix(irow, icol) = hist->GetBinContent(icol + 1, irow + 1);
        }
    }
    
    return matrix;
}

arma::Mat<double> HistToMatrix1D (const TH1* hist) {

    // Check type.
    assert(dynamic_cast<const TH1F*>(hist));

    // Initialise size variable(s).
    const unsigned N = hist->GetXaxis()->GetNbins();
    
    // Initialise output matrix (vector).
    arma::Mat<double> matrix (N, 1, arma::fill::zeros);
    
    // Fill matrix (vector) entries from histogram.
    for (unsigned i = 0; i < N; i++) {
        matrix(i, 0) = hist->GetBinContent(i + 1);
    }
    
    return matrix;
}

arma::Mat<double> HistFillMatrix (const TH1* hist) {
    
    // Determine dimension of matrix, and call appropriate specialised function.
    if (const TH2F* p = dynamic_cast<const TH2F*>(hist)) {
        return HistToMatrix2D(hist);
    } else {
        return HistToMatrix1D(hist);
    }
}

#endif // USE_ROOT

} // namespace
