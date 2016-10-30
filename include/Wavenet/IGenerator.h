#ifndef WAVELETML_IGenerator
#define WAVELETML_IGenerator

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
     * Defined derived classes to yield e.g. uniform, needle, gaussian, HepMC, or other custom input.
     **/

public:

    // Constructor(s).
    virtual void  IGenerator () {};

    // Destructor.
    virtual void ~IGenerator () = 0;

    // Method(s).
    virtual bool open  (const std::string& filename) = 0;
    virtual bool next  (arma::Mat<double>& input) = 0;
    virtual bool close () = 0;

private:

    // Data memeber(s).
    // ...

};

#endif