#ifndef WAVENET_GENERATORS_H
#define WAVENET_GENERATORS_H

/**
 * @file Generators.h
 * @author Andreas Sogaard
**/

// STL include(s).
#include <string>

// ROOT include(s).
// ...

// HepMC include(s).
// ...
 
// Armadillo include(s).
#include <armadillo>

// Wavenet include(s).
#include "Wavenet/IGenerator.h"


class NeedleGenerator : public IGenerator {

    /**
     * Derived class generating needle-like inut. 
    **/

public:

    // Constructor(s).
    NeedleGenerator () {}

    // Destructor.
    ~NeedleGenerator () {}

    // Method(s).
    inline bool next (arma::Mat<double>& input) {
       input.randu();
       input.elem( find(input < 0.995) ) *= 0;
       return true;
    }
    inline bool open  () { arma::arma_rng::set_seed_random(); return true; }
    inline bool close () { return true; }
    inline bool good  () { return true; }
    inline bool reset () { return close() && open(); }

private:

    // Data member(s).
    // ...
    
};

#endif