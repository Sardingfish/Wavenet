#ifndef WAVENET_GENERATORS_H
#define WAVENET_GENERATORS_H

/**
 * @file Generators.h
 * @author Andreas Sogaard
**/

// STL include(s).
#include <string>
#include <algorithm> /* std::max */

// HepMC include(s).
/* @TODO: Make dependent on installation */
#include "HepMC/GenEvent.h"
#include "HepMC/IO_GenEvent.h"
 
// Armadillo include(s).
#include <armadillo>

// Wavenet include(s).
#include "Wavenet/GeneratorBase.h"



namespace Wavenet {

enum class GeneratorMode {
   File     = 0,
   Uniform  = 1,
   Needle   = 2,
   Gaussian = 3
};

class NeedleGenerator : public GeneratorBase {

    /**
     * Derived class generating needle-like input. 
    **/

public:

    // Constructor(s).
    NeedleGenerator () { open(); }

    // Destructor.
    ~NeedleGenerator () {}

    // Method(s).
    inline const arma::Mat<double>& next () {
        // Check whether generator is properly initialised.
        if (!initialised()) {
            WARNING("Generator not properly initialised.");
            return _data;
        }

        // Generate next input matrix.
        _data.randu();
        _data.elem( find(_data < 0.995) ) *= 0;

        return _data;
    }

    inline bool open () { arma::arma_rng::set_seed_random(); return true; }
    
};


class UniformGenerator : public GeneratorBase {

    /**
     * Derived class generating uniformly random input. 
    **/

public:

    // Constructor(s).
    UniformGenerator () { open(); }

    // Destructor.
    ~UniformGenerator () {}

    // Method(s).
    inline const arma::Mat<double>& next () {
        // Check whether generator is properly initialised.
        if (!initialised()) {
            WARNING("Generator not properly initialised.");
            return _data;
        }
        
        // Generate next input matrix.
        _data.randu();

        return _data;
    }

    inline bool open () { arma::arma_rng::set_seed_random(); return true; }
    
};


class GaussianGenerator : public GeneratorBase {

    /**
     * Derived class generating gaussianly distributed input. 
    **/

public:

    // Constructor(s).
    GaussianGenerator () { open(); }

    // Destructor.
    ~GaussianGenerator () {}

    // Method(s).
    inline const arma::Mat<double>& next () {
        // Check whether generator is properly initialised.
        DEBUG("Entering");
        if (!initialised()) {
            ERROR("Generator not properly initialised. Exiting.");
            return _data;
        }

        // Generate next input matrix.
        _data.zeros();
        
        const unsigned int sizex = shape()[0];
        const unsigned int sizey = shape()[1];
        
        // -- Number of gaussian bumps to overlay
        const unsigned N = 1 + int(arma::as_scalar(arma::randu<arma::mat>(1,1)) * 3); // [1, 2, 3]
        
        for (unsigned i = 0; i < N; i++) {
            
            // -- Mean coordinates.
            int mux = int((arma::as_scalar(arma::randu<arma::mat>(1,1)) - 0.5) * (double)sizex);
            int muy = int((arma::as_scalar(arma::randu<arma::mat>(1,1)) - 0.5) * (double)sizey);

            // -- Axis coordinates.
            arma::Mat<double> hx = arma::linspace<arma::mat>(-((double)sizex-1)/2, ((double)sizex-1)/2, sizex);
            arma::Mat<double> hy = arma::linspace<arma::mat>(-((double)sizey-1)/2, ((double)sizey-1)/2, sizey);
            
            hx = repmat(hx, 1, sizey).t();
            hy = repmat(hy, 1, sizex);

            // -- Widths.
            double sx = std::max(arma::as_scalar(arma::randn<arma::mat>(1,1))*0.5 + sizex / 4., 2.); 
            double sy = std::max(arma::as_scalar(arma::randn<arma::mat>(1,1))*0.5 + sizey / 4., 2.); 
            arma::Mat<double> gauss = exp( - square(hx - mux) / (2*sq(sx)) - square(hy - mux) / (2*sq(sy)));
            
            // -- Normalise.
            gauss /= accu(gauss); // Unit integral.
            gauss *= 1./max(max(gauss));
            
            _data = _data + gauss.t();
            
        }
        DEBUG("Exiting.");
       return _data;
    }

    inline bool open () { arma::arma_rng::set_seed_random(); return true; }
    
};


class HepMCGenerator : public GeneratorBase {

    /**
     * Derived class generating input from HepMC files. 
    **/

public:

    // Constructor(s).
    //HepMCGenerator () {} // No empty constructor. 'filename' *must* be specified for this generator to make sense.
    HepMCGenerator (const std::string& filename) { open(filename); }

    // Destructor.
    ~HepMCGenerator () {
        if (_event) { delete _event; _event = NULL; }
        if (_IO)    { delete _IO;    _IO    = NULL; }
        if (_input) { delete _input; _input = NULL; }
    }

    // Method(s).
    inline const arma::Mat<double>& next () {
        DEBUG("Entering.");
        // Check whether generator is properly initialised.
        if (!initialised()) { /* @TODO: To be done externally, for GeneratorBase? */
            WARNING("Generator not properly initialised.");
            return _data;
        }

        // Generate next input matrix.
        _data.zeros();

        if (!good()) { /* @TODO: To be done externally, for GeneratorBase? */
            ERROR("Reader not good. Exiting.");
            return _data;
        }

        // -- Fill histogram.
        TH2F hist ("tmp", "", shape()[0], -3.2, 3.2, shape()[1], -PI, PI);
        
        HepMC::GenEvent::particle_const_iterator p     = _event->particles_begin();
        HepMC::GenEvent::particle_const_iterator p_end = _event->particles_end();
        

        for ( ; p != p_end; p++) {
            // -- Keep only (calorimeter) visible final state particles.
            if ((*p)->status() != 1) { continue; }
            unsigned idAbs = abs((*p)->pdg_id());
            if (idAbs == 12 || idAbs == 14 || idAbs == 16 || idAbs == 13) { continue; }
        
            hist.Fill((*p)->momentum().eta(), (*p)->momentum().phi(), (*p)->momentum().perp() / 1000.);
        }

        // Fill '_data' matrix with content from 'hist'.
        DEBUG("Calling HistFillMatrix.");
        HistFillMatrix(hist, _data);
        DEBUG("Done.");

        // -- Get next HepMC event.
        /**
         * This is done here, such that when HepMCGenerator::good is called, we're checking the *next* event, and therefore we don't risk generating a bad event.
        **/
        getNextHepMCEvent();
        DEBUG("Exiting.");
        return _data;
    }
    
    inline bool open  (const std::string& filename) { 
        DEBUG("Entering.");
        _filename = filename;
        open();
        DEBUG("Exiting.");
        return true; 
    }

    inline bool open  () { 
        DEBUG("Entering.");
        if (_filename == "") {
            ERROR("Filename not set. Exiting.");
            return false;
        }
        if (!fileExists(_filename)) {
            ERROR("File '%s' does not exist. Exiting.", _filename.c_str());
            return false;
        }
        
        _input = new std::fstream(_filename.c_str(), std::ios::in);
        _IO    = new HepMC::IO_GenEvent(*_input);
        DEBUG("Exiting.");
        return getNextHepMCEvent(); 
    }

    inline bool close () { 
        DEBUG("Entering.");
        if (_IO    != NULL) { delete _IO;    _IO    = NULL; }
        if (_input != NULL) { delete _input; _input = NULL; }
        DEBUG("Exiting.");
        return true; 
    }

    inline bool good  () { 
        DEBUG("Entering.");
        if (!_IO)    { INFO("Member object '_IO' is null."); }
        if (!_event) { INFO("Member object '_event' is null."); }
        DEBUG("Exiting.");
        return _IO && _event; 
     }

    inline bool getNextHepMCEvent () {
        DEBUG("Entering.");
        if (_event) { delete _event; _event = 0; }
        if (_IO)    { _event = _IO->read_next_event(); } else {
            WARNING("Member object '_IO' is null.");
            DEBUG("Exiting.");
            return false;
        }
        DEBUG("Exiting.");
        return true;
    }
    
private:

    // Data member(s).
    std::string         _filename = "";
    std::istream*       _input = nullptr;
    HepMC::IO_GenEvent* _IO    = nullptr;
    HepMC::GenEvent*    _event = nullptr;
    
};

} // namespace

#endif // WAVENET_GENERATORS_H