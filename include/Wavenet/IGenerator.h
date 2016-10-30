#ifndef WAVENET_IGENERATOR_H
#define WAVENET_IGENERATOR_H

/**
 * @file IGenerator.h
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
// ...


class IGenerator {

    /**
     * Interface class for input generators. 
     *
     * Define derived classes to yield e.g. uniform, needle, gaussian, HepMC, or other custom input.
    **/

public:

    // Destructor.
    virtual ~IGenerator () {}

    // Method(s).
    virtual bool next  (arma::Mat<double>& input) = 0;
    virtual bool open  () = 0;
    virtual bool close () = 0;
    virtual bool good  () = 0;
    virtual bool reset () = 0;

private:

    // Data member(s).
    // ...

};

#endif