#ifndef WAVELETML_Utils
#define WAVELETML_Utils

// STL include(s).
#include <cmath>
#include <sys/stat.h> /* struct stat */

// ROOT include(s).
#include "TH2.h"

// Armadillo include(s).
#include <armadillo>

using namespace arma;

const double EPS = 1e-12;
const double PI = 3.14159265359;

// Determine wether the given number is radix 2, i.e. satisfies:y 2^x in Naturals
inline bool isRadix2(const unsigned& x) { return (unsigned(log2(x)) % 1) == 0; }

// Square of number.
template<class T>
inline T sq(const T& x) { return x * x; }


// Absolute value of number.
template<class T>
inline T norm(const T& x) { return sqrt(sq(x)); }

// Sign of number
template<class T>
inline T sign(const T& x) { return x == 0 ? 0 : x/norm(x); }

template<class T>
inline arma::Col<T> sign(const arma::Col<T>& v) { arma::Col<T> s = v; return s.transform( [](double x) { return sign(x); } ); }

template<class T>
inline arma::Mat<T> sign(const arma::Mat<T>& M) { return M / abs(M + EPS);  }

template<class T>
inline arma::Mat<T> onset(const arma::Mat<T>& M) { arma::Mat<T> o = M; return o.transform( [](double x) { return (x > 0 ? 1 : 0); } );  }


// Shift arma::Row<double> by specified amount.
/*
void rowshift (arma::Row<double>& row, const int& shift) {
    
    unsigned length = row.n_elem;
    
    if (shift > 0 && shift < length) {
        
        arma::Row<double> remainder (shift);
        remainder = row( span(length - shift, length - 1) );
        row( span(shift, length - 1) ) = row( span(0, length - 1 - shift) );
        row( span(0, shift - 1) ) = remainder;
        
    } else {
        
        if (length <  abs(shift)) {
            rowshift(row, sign(shift) * (abs(shift) % length));
        }
        
        if (shift  <  0) {
            rowshift(row, length + shift);
        }
        
    }
    
    return;
    
}
 */

// Rowshift, given an arma matrix subview as input.
/*
arma::Row<double> rowshift(arma::subview_row<double> row, const int& shift) {
    arma::Row<double> newRow = row;
    rowshift(newRow, shift);
    return newRow;
}
 */

// Convert arma matrix to ROOT TH2.
inline TH2F MatrixToHist (const arma::Mat<double>& matrix, const double& range) {
    
    unsigned N1 = size(matrix, 0), N2 = size(matrix, 1);
    
    TH2F hist ("hist", "", N1, -range, range, N2, -range, range);
    
    for (unsigned i = 0; i < N1; i++) {
        for (unsigned j = 0; j < N2; j++) {
            hist.SetBinContent(j + 1, i + 1, matrix(i,j));
        }
    }
    
    return hist;
}

// Convert ROOT TH2 to arma matrix .
inline arma::Mat<double> HistToMatrix (const TH2F& hist) {
    
    unsigned N1 = hist.GetYaxis()->GetNbins(), N2 = hist.GetXaxis()->GetNbins();
    
    arma::Mat<double> matrix (N1, N2, fill::zeros);
    
    for (unsigned i = 0; i < N1; i++) {
        for (unsigned j = 0; j < N2; j++) {
            matrix (i,j) = hist.GetBinContent(j + 1, i + 1);
        }
    }
    
    return matrix;
}

// Randomly generate points on N-sphere.
inline arma::Col<double> PointOnNSphere (const unsigned& N, const double& rho = 0., bool restrict = false) {
    
    arma::Col<double> coords (N, fill::ones);
    
    coords = randn< arma::Col<double> > (N);
    
    coords *= (1 + arma::as_scalar(randn(1)) * rho) / arma::norm(coords);
    
    if (restrict) { // Require first coordinate to be the largest one, and positive.
        cout << "<PointOnNSphere> Starting from:" << endl << coords;
        if (std::abs(coords.at(0)) < std::abs(coords.at(N - 1))) {
            coords = arma::flipud(coords);
            cout << "<PointOnNSphere> Flipping to:" << endl << coords;
        }
        if (coords.at(0) < 0) {
            coords *= -1;
            cout << "<PointOnNSphere> Changing sign to:" << endl << coords;
        }
        cout << "<PointOnNSphere> Done:" << endl;
    }
    
    return coords;
}


// Check whether file exists.
inline bool fileExists (const std::string& filename) {
    ifstream f(filename.c_str());
    bool exists = f.good();
    f.close();
    return exists;
}

inline bool dirExists (const std::string& dir) {
    struct stat statbuf;
    bool exists = false;
    if (stat(dir.c_str(), &statbuf) != -1) {
        if (S_ISDIR(statbuf.st_mode)) { exists = true; }
    }
    return exists;
}

// Check whether string contains only numeric characters.
inline bool isNumber (const std::string& s)
{
    return !s.empty() && std::find_if(s.begin(), s.end(), [](char c) { return !(std::isdigit(c) || strcmp(&c, ".") == 0 || (strcmp(&c, " ") == 0)); }) == s.end();
}

inline bool isEmpty (const std::string& s)
{
    return !s.empty() && std::find_if(s.begin(), s.end(), [](char c) { return !(strcmp(&c, " ") == 0); }) == s.end();
}

#endif
