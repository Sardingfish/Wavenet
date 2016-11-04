#include "Wavenet/Utilities.h"
#include "Wavenet/Wavenet.h" /* To create Wavenet instance. */ 

namespace wavenet {

/// Armadillo-specific functions.
arma::Col<double> PointOnNSphere (const unsigned& N, const double& rho, bool restrict) {
    
    arma::Col<double> coords (N, arma::fill::ones);
    
    coords = arma::randn< arma::Col<double> > (N);

    arma::arma_rng::set_seed_random();
    coords *= (1 + arma::as_scalar(arma::randn(1)) * rho) / arma::norm(coords);
    
    if (restrict) { // Require first coordinate to be the largest one, and positive.
        FCTINFO("Starting from:");
        std::cout << coords << std::endl;
        if (std::abs(coords.at(0)) < std::abs(coords.at(N - 1))) {
            coords = arma::flipud(coords);
            FCTINFO("Flipping to:");
            std::cout << coords << std::endl;
        }
        if (coords.at(0) < 0) {
            coords *= -1;
            FCTINFO("Changing sign to:");
            std::cout << coords << std::endl;
        }
        FCTINFO("Done");
    }
    
    return coords;
}


/// ROOT-specific functions.
#ifdef USE_ROOT

TGraph costGraph (const std::vector< double >& costLog) {
    
    const unsigned N = costLog.size();
    double x[N], y[N];
    for (unsigned i = 0; i < N; i++) {
        x[i] = i;
        y[i] = costLog.at(i);
        
    }
    
    TGraph graph (N, x, y);
    
    return graph;
}


TGraph costGraph (const std::vector< arma::Col<double> >& filterLog, const std::vector< arma::Mat<double> >& X) {
    
    Wavenet wavenet;
    const unsigned N = filterLog.size();
    double x[N], y[N];
    for (unsigned i = 0; i < N; i++) {
        x[i] = i;
        y[i] = 0;
        wavenet.setFilter(filterLog.at(i));
        for (const arma::Mat<double>& x : X) {
            y[i] += wavenet.cost(wavenet.coeffsFromActivations(wavenet.forward(x)));
        }
        y[i] /= (double) X.size();
    }
    
    TGraph graph (N, x, y);
    
    return graph;
}


std::unique_ptr<TH1> MatrixToHist2D (const arma::Mat<double>& matrix, const double& range) {
    /**
     * Return a unique pointer to a ROOT TH2F filled with the content of the arma matrix 'matrix'.
    **/
    
    unsigned N1 = size(matrix, 0), N2 = size(matrix, 1);
    
    std::unique_ptr<TH1> hist (new TH2F("hist", "", N1, -range, range, N2, -range, range));
    
    for (unsigned i = 0; i < N1; i++) {
        for (unsigned j = 0; j < N2; j++) {
            hist->SetBinContent(i + 1, j + 1, matrix(i,j));
        }
    }
    
    return hist;
}

std::unique_ptr<TH1> MatrixToHist1D (const arma::Mat<double>& matrix, const double& range) {
    /**
     * Return a unique pointer to a ROOT TH1F filled with the content of the arma matrix 'matrix'.
    **/

    unsigned N1 = size(matrix, 0);
    
    std::unique_ptr<TH1> hist (new TH1F("hist", "", N1, -range, range) );
    
    for (unsigned i = 0; i < N1; i++) {
        hist->SetBinContent(i + 1, matrix(i,0));
    }
    
    return hist;
}

std::unique_ptr<TH1> MatrixToHist (const arma::Mat<double>& matrix, const double& range) {
    /**
     * Determine the appropriate dimension of data, and return a unique pointer to a TH1-type object (either a TH1F or a TH2F).
    **/

    if (size(matrix,1) == 1) {
        return MatrixToHist1D(matrix, range);
    } else {
        return MatrixToHist2D(matrix, range);
    }
    return nullptr;
}

arma::Mat<double> HistFillMatrix2D (const TH1* hist, arma::Mat<double>& matrix) {
    
    assert(dynamic_cast<const TH2F*>(hist));

    unsigned N1 = hist->GetYaxis()->GetNbins(), N2 = hist->GetXaxis()->GetNbins();
    
    assert(N1 = size(matrix,0));
    assert(N2 = size(matrix,1));

    matrix.zeros();
    
    for (unsigned i = 0; i < N1; i++) {
        for (unsigned j = 0; j < N2; j++) {
            matrix (i,j) = hist->GetBinContent(i + 1, j + 1);
        }
    }
    
    return matrix;
}

arma::Mat<double> HistFillMatrix1D (const TH1* hist, arma::Mat<double>& matrix) {
    
    assert(dynamic_cast<const TH1F*>(hist));

    unsigned N1 = hist->GetYaxis()->GetNbins();
    
    assert(N1 = size(matrix,0));
    
    matrix.zeros();
    
    for (unsigned i = 0; i < N1; i++) {
        matrix (i,0) = hist->GetBinContent(i + 1);
    }
    
    return matrix;
}

arma::Mat<double> HistFillMatrix (const TH1* hist, arma::Mat<double>& matrix) {
    /**
     * Determine the appropriate dimension of data, and return a unique pointer to a TH1-type object (either a TH1F or a TH2F).
    **/

    if (size(matrix,1) == 1) {
        return HistFillMatrix1D(hist, matrix);
    } else {
        return HistFillMatrix2D(hist, matrix);
    }
}

arma::Mat<double> HistToMatrix2D (const TH1* hist) {
    
    assert(dynamic_cast<const TH2F*>(hist));

    unsigned N1 = hist->GetYaxis()->GetNbins(), N2 = hist->GetXaxis()->GetNbins();
    
    arma::Mat<double> matrix (N1, N2, arma::fill::zeros);
    
    for (unsigned i = 0; i < N1; i++) {
        for (unsigned j = 0; j < N2; j++) {
            matrix (i,j) = hist->GetBinContent(j + 1, i + 1);
        }
    }
    
    return matrix;
}

arma::Mat<double> HistToMatrix1D (const TH1* hist) {

    assert(dynamic_cast<const TH1F*>(hist));

    unsigned N1 = hist->GetXaxis()->GetNbins();
    
    arma::Mat<double> matrix (N1, 1, arma::fill::zeros);
    
    for (unsigned i = 0; i < N1; i++) {
        matrix (i,0) = hist->GetBinContent(i + 1);
    }
    
    return matrix;
}


arma::Mat<double> HistFillMatrix (const TH1* hist) {
    /**
     * Determine the type of the histogram, and return an appropriately filled Armadillo matrix.
    **/

    if (const TH2F* p = dynamic_cast<const TH2F*>(hist)) {
        return HistToMatrix2D(hist);
    } else {
        return HistToMatrix1D(hist);
    }
}

#endif // USE_ROOT

} // namespace
