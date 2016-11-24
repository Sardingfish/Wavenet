#include "Wavenet/CostFunctions.h"

namespace wavenet {


double SparseTerm (const arma::Col<double>& c) {
    
    /**
     * For wavelet coefficients {c} ordered by increasing absolute value
     * (|c_{i}| < |c_{i+1}|) the Gini coefficient can be written as 
     * G({c}) = f({c})/g({c}) with:
     *   f({c}) = \sum_{i = 0}^{N - 1} (2 i - N - 1) |c_{i}|
     *   g({c})  = N * \sum_{i = 0}^{N - 1} |c_{i}|
     * where N is the number of wavelet coefficients, N = |{c}|.
     *
     * Since, by this definition, large values means a very inequal (sparse) 
     * distribution and low value means a very equal (non-sparse) distribution, 
     * we choose to use 1 - G({c}) as our sparsity metric, since the 
     * minimisation then leads to sparse dsitributions.
     */

    // Initialise the number of wavelet coefficients. 
    const int N = c.n_elem;
    
    // Initialise vector of wavelet coefficients sorted by absoute value.
    arma::Col<double> cSortAbs = sort(abs(c));

    // Initialise vector of summation indices.
    arma::Row<double> indices  = arma::linspace< arma::Row<double> >(1, N, N);
    
    // Compute numerator (f) and denominator (g) terms.
    double f = dot( (2*indices - N - 1), cSortAbs);
    double g = N * sum(cSortAbs);
    
    // Return sparsity measure.
    return 1. - f/g;
}

double SparseTerm (const arma::Mat<double>& C) {

    // Vectoris input matrix.
    const arma::Col<double> c = vectorise(C);

    // Get sparsity term on vector form.
    return SparseTerm(c);
}


arma::Col<double> SparseTermDeriv (const arma::Col<double>& c) {
   
     /**
     * For wavelet coefficients {c} ordered by increasing absolute value
     * (|c_{i}| < |c_{i+1}|) the Gini coefficient can be written as 
     * G({c}) = f({c})/g({c}) with:
     *   f({c}) = \sum_{i = 0}^{N - 1} (2 i - N - 1) |c_{i}|
     *   g({c})  = N * \sum_{i = 0}^{N - 1} |c_{i}|
     * where N is the number of wavelet coefficients, N = |{c}|.
     *
     * Using 1 - G({c)) as the sparsity metric, the derviative is then:
     *   delta(i) = \frac{d}{da_{i}}( 1 - G({a}) ) = - (f'*g - f*g') / g^2
    **/
        
    // Initialise the number of wavelet coefficients.
    const int N = c.n_elem;

    // Initialise vector of absolute wavelet coefficient values.
    arma::Col<double> cAbs  = abs(c);

    // Initialise vector of wavelet coefficient signs, such that 
    // c = cAbs % cSign.
    arma::Col<double> cSign = sign(c);

    // Get indices of wavelet coefficient sorted by ascending absolute value
    arma::uvec idxSortAbs = sort_index(cAbs, "ascend");


    // Initialise vector of wavelet coefficients sorted by ascending absolute 
    // value.
    arma::Col<double> cSortAbs = cAbs.elem(idxSortAbs);

    // Initialise vector of summation indices.
    arma::Col<double> indices  = arma::linspace< arma::Col<double> >(1, N, N);
    
    // Compute numerator (f) and denominator (g) terms in the Gini coefficient.
    double f = dot( (2*indices - N - 1), cSortAbs );
    double g = N * sum(cSortAbs);
    
    // Compute df/d|c| for sorted {c}.
    arma::Col<double> dfSort = (2*indices - N - 1);
    
    // Compute df/d|c| for {c} with original ordering.
    arma::Col<double> df (N, arma::fill::zeros);
    df.elem(idxSortAbs) = dfSort;
    
    // Compute df/dc for {c} with original ordering.
    df = df % cSign; // Elementwise multiplication.
    
    // Compute dg/dc for {c} with original ordering.
    arma::Col<double> dg = N * cSign;
    
    // Compute combined derivative.
    arma::Col<double> gradient = - (df * g - f * dg)/pow(g, 2.);
    
    return gradient;
}

arma::Mat<double> SparseTermDeriv (const arma::Mat<double>& C) {

    // Vectorise matrix input.
    arma::Col<double> c = vectorise(C);

    // Get sparsity term derivative on vector form.
    arma::Col<double> Gradient = SparseTermDeriv(c);

    // Reshape output vector to original matrix shape.
    return reshape(Gradient, size(C));
}

double RegTerm (const arma::Col<double>& a, const bool& doWavelet) {
    
    // Initialise number of filter coefficients.
    const int N = a.n_elem;


    // Define Kroenecker delta vector, with zeros everywhere except central 
    // entry, which is one
    arma::Col<double> kroenecker_delta (N+1, arma::fill::zeros);
    kroenecker_delta(N/2) = 1;
    
    // Compute regularisation term by term.
    double R = 0.; 


    /**
     * (C2): Orthogonality of scaling functions
     *
     * Mathematical condition:
     *   \sum_{k} a_{k} a_{k + 2m} = #delta_{0,m} \quad \forall m \in \mathbb{Z}
     */
    float R2 = 0.;
    // Perform sum over index m. Summation range taken to be [-N/2, N/2] since 
    // all other values will always result in a_{k + 2m} == 0 for k in [0,N).
    for (const int& m : arma::linspace(-N/2, N/2, N + 1)) {

        // Initialise Kroenecker delta value for the current value of m.
        float delta = (m == 0 ? 1 : 0);

        // Perform sum over index k.
        float term = 0.;
        for (unsigned k = 0; k < N; k++) {

            // Add term a_{k}: a_{k + 2m}
            if (a.in_range(k + 2*m)) {
                term += a(k) * a(k + 2*m);
            }            
        }

        // Take squared deviation from Kroenecker delta.
        R2 += sq(term - delta);
    }
    R += R2;

    // Wavelet-specific regularisation terms.
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
        float R3 = 0.;
        // Perform sum over index m. Summation range taken to be [-N/2, N/2] since 
        // all other values will always result in b_{k + 2m} == 0 for k in [0,N).
        for (const int& m : arma::linspace(-N/2, N/2, N + 1)) {

            // Initialise Kroenecker delta value for the current value of m.
            float delta = (m == 0 ? 1 : 0);

            // Perform sum over index k.
            float term = 0.;
            for (unsigned k = 0; k < N; k++) {

                // Add term:  b_{k} b_{k + 2m}
                if (b.in_range(k + 2*m)) {
                    term += b(k) * b(k + 2*m);
                }            
            }

            // Take squared deviation from Kroenecker delta.
            R3 += sq(term - delta);
        }
        R += R3;

        
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
         * 
         * Note:
         *   This condition is automatically satisfied by the definition of 
         *     b_{k} = (-1)^{k} a_{N - k - 1}
         *   which is why it is not imposed explicitly.
         */
        /*
        float R5 = 0.;
        // Perform sum over index m. Summation range taken to be [-N/2, N/2] since 
        // all other values will always result in b_{k + 2m} == 0 for k in [0,N).
        for (const int& m : arma::linspace(-N/2, N/2, N + 1)) {

            // Perform sum over index k.
            float term = 0.;
            for (unsigned k = 0; k < N; k++) {

                // Add term a_{k}: a_{k + 2m}
                if (b.in_range(k + 2*m)) {
                    term += a(k) * b(k + 2*m);
                }            
            }

            // Take squared deviation from Kroenecker delta.
            R5 += sq(term - 0);
        }
        R += R5;
        */
        
    }
    
    return R / 2.;
}

arma::Col<double> RegTermDeriv (const arma::Col<double>& a, const bool& doWavelet) {
    
    /**
     * Taking the derivative of each term in the sum of squared deviations from 
     * the Kroeneker delta, in the regularisation objective function.
     *
     * Since the regularisation terms are generally on the form (...)^2, the 
     * corresponding derivative will be on the form 2 x (...) x d/da (...)
     */ 

    // Initialise number of filter coefficients.
    const int N = a.n_elem;

    // Initialise filter coefficient gradient vector.
    arma::Col<double> gradient (N, arma::fill::zeros); 


    /**
     * (C2): Orthogonality of scaling functions
     *
     * Mathematical expression:
     *   \nabla R_{2}(\{a\}) = \hat{e}_{i} \times \sum_{m} 2 \times 
     *                         ([\sum_{k} a_{k} a_{k + 2m}] - \delta_{m,0}) 
     *                         \times (a_{i + 2m} + a_{i - 2m})
     */
    // Loop over summation index m. Summation range taken to be [-N/2, N/2] 
    // since all other values will always result in a_{k + 2m} == 0 for k in 
    // [0,N).
    for (const int& m : arma::linspace(-N/2, N/2, N + 1)) {

        // Initialise pre-factor.
        double prefactor = 0.;

        // Loop over summation index k.
        for (int k = 0; k < N; k++) {
            if (a.in_range(k + 2 * m)) {
                prefactor += a(k) * a(k + 2 * m);
            }
        }

        // Substract kroenecker delta: \delta_{0,m}
        prefactor -= (m == 0 ? 1. : 0);

        // Multiply by lowered power of 2.
        prefactor *= 2.;
    
        // Initialise inner derivative vector
        arma::Col<double> inner_derivative (N, arma::fill::zeros);

        // Loop over filter coefficient index.
        for (int i = 0; i < N; i++) {
            if (a.in_range(i + 2 * m)) { 
                inner_derivative(i) += a(i + 2 * m);
            }
            if (a.in_range(i - 2 * m)) {
                inner_derivative(i) += a(i - 2 * m);
            }
        }
        
        // Add term to total gradient.
        gradient += prefactor * inner_derivative;
    }


    // Wavelet-specific regularisation terms.
    if (doWavelet) {

        // Compute high-pass filter coefficients.
        arma::Col<double> b     (N, arma::fill::zeros);
        arma::Col<double> bsign (N, arma::fill::zeros);
        for (unsigned i = 0; i < N; i++) {
            b    (i) = pow(-1, i) * a(N - i - 1);
            bsign(i) = pow(-1, N - i - 1); // Used in (C4).
        }
        

        /**
         * (C1): Dilation equation.
         *
         * Mathematical expression:
         *   \nabla R_{1}(\{a\}) = \hat{e}_{i} \times 2 \times 
         *                         ([\sum_{k} a_{k}] - \sqrt{2})
         */
        gradient += 2 * (sum(a) - sqrt(2)) * arma::Col<double> (N, arma::fill::ones);
        

        /**
         * (C3): Orthonormality of wavelet functions.
         *
         * Mathematical expression:
         *   \nabla R_{3}(\{a\}) = \hat{e}_{i} \times \sum_{m} 2 \times 
         *                         ([\sum_{k} b_{k} b_{k + 2m}] - \delta_{m,0}) 
         *                         \times (a_{i + 2m} + a_{i - 2m})
         */
        // Loop over summation index m. Summation range taken to be [-N/2, N/2] 
        // since all other values will always result in b_{k + 2m} == 0 for k in
        // [0,N).
        for (const int& m : arma::linspace(-N/2, N/2, N + 1)) {

            // Initialise pre-factor.
            double prefactor = 0.;

            // Loop over summation index k.
            for (int k = 0; k < N; k++) {
                if (b.in_range(k + 2 * m)) {
                    prefactor += b(k) * b(k + 2 * m);
                }
            }

            // Substract kroenecker delta: \delta_{0,m}
            prefactor -= (m == 0 ? 1. : 0);

            // Multiply by lowered power of 2.
            prefactor *= 2.;
        
            // Initialise inner derivative vector
            arma::Col<double> inner_derivative (N, arma::fill::zeros);

            // Loop over filter coefficient index.
            for (int i = 0; i < N; i++) {
                if (a.in_range(i + 2 * m)) { 
                    inner_derivative(i) += a(i + 2 * m);
                }
                if (a.in_range(i - 2 * m)) {
                    inner_derivative(i) += a(i - 2 * m);
                }
            }
            
            // Add term to total gradient.
            gradient += prefactor * inner_derivative;
        }

        
        /**
         * (C4): High-pass filter.
         *
         * Mathematical expression:
         *   \nabla R_{4}(\{a\}) = \hat{e}_{i} \times \sum_{m} 2 \times 
         *                         ([\sum_{k} b_{k}]) \times (-1)^{N - i - 1}
         */
        gradient += 2 * (sum(b) - 0) * bsign;
        

        /**
         * (C5): Orthogonality of scaling and wavelet functions.
         *
         * The corresponding condition is automatically satisfied by the 
         * definition of 
         *     b_{k} = (-1)^{k} a_{N - k - 1}
         * which means that the derivative is identically equal to zero.
         */
        // gradient += arma::Col<double> (N, arma::fill::zeros);

    }
    
    return gradient;
}

} // namespace
