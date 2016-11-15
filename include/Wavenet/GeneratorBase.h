#ifndef WAVENET_GENERATORBASE_H
#define WAVENET_GENERATORBASE_H

/**
 * @file   GeneratorBase.h
 * @author Andreas Sogaard
 * @date   14 November 2016
 * @brief  Mix-in class for input generators. 
 */

// STL include(s).
#include <string> /* std::string */
#include <vector> /* std::vector */

// Armadillo include(s).
#include <armadillo>

// Wavenet include(s).
#include "Wavenet/Utilities.h"
#include "Wavenet/Logger.h"


namespace wavenet {

/**
 * Mix-in class for input generators. 
 *
 * The GeneratorBase provides functionality to set the shape of the generated 
 * input as well as a few accessor/mutator functions. Define derived classes to 
 * yield e.g. uniform, needle, gaussian, HepMC, or other custom input. 
 */
class GeneratorBase : public Logger {

public:

    /// Destructor.
    virtual ~GeneratorBase () {}


    /// Generator method(s).
    // These are the main methods for any class deriving from GeneratorBase. 
    // Purely virtual methods must be implemented in the derived, specialised 
    // generators. The remaining virtual methods can be overwritten.

    // Get the next input from the generator.
    virtual const arma::Mat<double>& next  () = 0;

    // Whether the generator is in a good condition, i.e. whether we can safely 
    // produce the next generator input.
    virtual bool good  () const = 0;

    // Open the generator.
    virtual bool open  ();

    // Close the generator.
    virtual bool close ();

    // Reset the generator.
    inline  bool reset () { return close() && open(); }


    /// Shape method(s).
    // Set the shape of the generator input.
    bool setShape (const std::vector<unsigned>& shape);


    /// Get method(s)
    // Get the shape of the generator input.
    std::vector<unsigned> shape () const { return m_shape; }

    // Get dimension of generator input. Currently supports up to two-
    // dimensional input.
    inline unsigned dim () const { return m_shape[1] == 1 ? 1 : 2; }

    // Whether the current GeneratorBase instance is properly intialised.
    inline bool initialised () const { return m_initialised; }


protected:

    /// Internal method(s).
    // Resize the generator input matrix to accorrding to the current, internal, 
    // accepted shape
    bool resize_ ();

    // Called during each 'next' function call, to check whether the generator 
    // is properly configured.
    bool check_ ();


protected:

    /// Data member(s).
    // Whether the generator is properly initialised.
    bool m_initialised = false;

    // The shape of the generator input.
    std::vector<unsigned> m_shape = {}; 

    // Armadillo matrix, holding the input produced by the generator.
    arma::Mat<double> m_data = {};

};

} // namespace

#endif // WAVENET_GENERATORBASE_H