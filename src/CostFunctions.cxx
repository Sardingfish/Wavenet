#include "Wavenet/CostFunctions.h"

namespace wavenet {


double SparseTerm (const arma::Col<double>& y) {
    
    /**
     * Using the Gini coefficient as the metric for sparsity.
     */

    const int N = y.n_elem;
    
    arma::Col<double> ySortAbs = sort(abs(y));
    arma::Row<double> indices  = arma::linspace< arma::Row<double> >(1, N, N);
    
    double f = - dot( (2*indices - N - 1), ySortAbs);
    double g = N * sum(ySortAbs);
    
    return f/g + 1;
}

double SparseTerm (const arma::Mat<double>& Y) {
    const arma::Col<double> y = vectorise(Y);
    return SparseTerm(y);
}


arma::Col<double> SparseTermDeriv (const arma::Col<double>& y) {
   
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

arma::Mat<double> SparseTermDeriv (const arma::Mat<double>& Y) {
    arma::Col<double> y = vectorise(Y);
    arma::Col<double> D = SparseTermDeriv(y);
    return reshape(D, size(Y));
}

double RegTerm (const arma::Col<double>& a, const bool& doWavelet) {
    
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
    if (doWavelet) {

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
    
    return R / 2.;
}

arma::Col<double> RegTermDeriv (const arma::Col<double>& a, const bool& doWavelet) {
    
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
    if (doWavelet) {

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
    
    return D/2.;
}

} // namespace
