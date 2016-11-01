#ifndef WAVENET_GENERATORBASE_H
#define WAVENET_GENERATORBASE_H

/**
 * @file GeneratorBase.h
 * @author Andreas Sogaard
**/

// STL include(s).
#include <string> /* std::string */
#include <vector> /* std::vector */

// Armadillo include(s).
#include <armadillo>

// Wavenet include(s).
#include "Wavenet/Utils.h"
#include "Wavenet/Logger.h"


class GeneratorBase : public Logger {

    /**
     * Abstract class for input generators. 
     *
     * Define derived classes to yield e.g. uniform, needle, gaussian, HepMC, or other custom input.
    **/

public:

    // Destructor.
    virtual ~GeneratorBase () {}

    // Method(s).
    virtual const arma::Mat<double>& next  () = 0;
    virtual bool open  ();
    virtual bool close ();
    virtual bool good  ();
    inline  bool reset ();

    bool setShape (const std::vector<unsigned int>& shape);

    std::vector<unsigned int> shape ();

    unsigned int dim ();

    bool initialised ();


private:

    // Internal method(s).
    bool _resize ();


protected:

    // Data member(s).
    bool _initialised = false;
    std::vector<unsigned int> _shape;
    arma::Mat<double> _data;

};

#endif