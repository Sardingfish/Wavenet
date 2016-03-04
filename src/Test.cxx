// STL include(s).
#include <string>
#include <vector>

// ROOT include(s).

// WaveletML include(s).
#include <armadillo>
#include "WaveletML.h"
#include "LowpassOperator.h"

using namespace arma;

int main (int argc, char* argv[]) {
    cout << "Running WaveletML test." << endl;
    
    // Timing.
    /*
    wall_clock timer;
    unsigned N = 100; // 10000000;
    
    timer.tic();
    for (unsigned i = 0; i < N; i++) {
            LowpassOperator op({1,2}, 2);
    }
    double elapsed = timer.toc();

    cout << "number of seconds: " << elapsed << " / " << N << endl;

    LowpassOperator op1({1,2}, 2);
    op1.print();
     */
    Col<double> filter ({1, -2, 3, 4});
    
    WaveletML ML;
    ML.setFilter(filter);
    ML.cacheOperators(2);
    ML.cacheWeights(2);
    
    /*
    cout << "  Highpass operator to filter:" << endl;
    filter.print();
    cout << "  ... is:" << endl;
    ML._cachedHighpassOperators (1, 0).print();
    cout << " -- " << endl;
    ML._cachedHighpassOperators (0, 0).print();
    cout << " -- " << endl;
    for (unsigned i = 0; i < filter.n_elem; i++) {
        cout << "  weight (" << i << "):" << endl;
        ML._cachedHighpassWeights(0,i).print();
    }
    */
    
    cout << "Done." << endl;
    
    return 1;
}
