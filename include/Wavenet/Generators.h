#ifndef WAVENET_GENERATORS_H
#define WAVENET_GENERATORS_H

/**
 * @file   Generators.h
 * @author Andreas Sogaard
 * @date   14 November 2016
 * @brief  Various derived generator classes.
 */

// STL include(s).
#include <string> /* std::string */
#include <algorithm> /* std::max */
#include <memory> /* std::unique_ptr */
#include <utility> /* std::move */

#ifdef USE_HEPMC
// HepMC include(s).
#include "HepMC/GenEvent.h"
#include "HepMC/IO_GenEvent.h"
#endif // USE_HEPMC

// Armadillo include(s).
#include <armadillo>

// Wavenet include(s).
#include "Wavenet/GeneratorBase.h"


namespace wavenet {

/**
 * Derived class generating needle-like input. 
 */
class NeedleGenerator : public GeneratorBase {

public:

    /// Constructor(s).
    NeedleGenerator () { open(); }


    /// Destructor.
    ~NeedleGenerator () {}


    /// Generator method(s).
    virtual inline const arma::Mat<double>& next () {

        // Check whether generator is properly set up.
        check_();

        // Generate next input matrix.
        do {
            m_data.randu();
            m_data.elem( find(m_data < 0.995) ) *= 0;
        } while (arma::accu(m_data) == 0);

        return m_data;
    }

    virtual inline bool good () const { return true; }

    virtual inline bool open () { arma::arma_rng::set_seed_random(); return true; }
  
};


/**
 * Derived class generating uniformly random input. 
 */
class UniformGenerator : public GeneratorBase {

public:

    /// Constructor(s).
    UniformGenerator () { open(); }


    /// Destructor.
    ~UniformGenerator () {}


    /// Generator method(s).
    virtual inline const arma::Mat<double>& next () {

        // Check whether generator is properly set up.
        check_();
        
        // Generate next input matrix.
        m_data.randu();

        return m_data;
    }

    virtual inline bool good () const { return true; }

    virtual inline bool open () { arma::arma_rng::set_seed_random(); return true; }
    
};


/**
 * Derived class generating gaussianly distributed input. 
*/
class GaussianGenerator : public GeneratorBase {


public:

    /// Constructor(s).
    GaussianGenerator () { open(); }


    /// Destructor.
    ~GaussianGenerator () {}


    /// Generator method(s).
    virtual inline const arma::Mat<double>& next () {

        // Check whether generator is properly set up.
        check_();

        // Initialise generator input to zeros.
        m_data.zeros();
        
        // Initialise size variables.
        const unsigned sizex = shape()[0];
        const unsigned sizey = shape()[1];
        
        // Get number of gaussian bumps to overlay.
        const unsigned N = 1 + int(arma::as_scalar(arma::randu<arma::mat>(1,1)) * 3); // [1, 2, 3]
        
        // Generate each bump separately.
        for (unsigned i = 0; i < N; i++) {
            
            // Mean coordinates.
            int mux = int((arma::as_scalar(arma::randu<arma::mat>(1,1)) - 0.5) * (double)sizex);
            int muy = int((arma::as_scalar(arma::randu<arma::mat>(1,1)) - 0.5) * (double)sizey);

            // Axis coordinates: Matrices of x- and y-coordinates, resp.
            arma::Mat<double> hx = arma::linspace<arma::mat>(-((double)sizex-1)/2, ((double)sizex-1)/2, sizex);
            arma::Mat<double> hy = arma::linspace<arma::mat>(-((double)sizey-1)/2, ((double)sizey-1)/2, sizey);
            
            hx = repmat(hx, 1, sizey).t();
            hy = repmat(hy, 1, sizex);

            // Widths.
            double sx = std::max(arma::as_scalar(arma::randn<arma::mat>(1,1))*0.5 + sizex / 4., 2.); 
            double sy = std::max(arma::as_scalar(arma::randn<arma::mat>(1,1))*0.5 + sizey / 4., 2.); 
            arma::Mat<double> gauss = exp( - square(hx - mux) / (2*sq(sx)) - square(hy - muy) / (2*sq(sy)));
            
            // Normalise to unit height
            gauss /= accu(gauss);
            gauss *= 1./max(max(gauss));
            
            // Add to generator input matrix.
            m_data = m_data + gauss.t();
            
        }
        
        return m_data;
    }

    virtual inline bool good () const { return true; }

    virtual inline bool open () { arma::arma_rng::set_seed_random(); return true; }
    
};


#ifdef USE_HEPMC
/**
 *  Derived class generating input from HepMC files. 
 */
class HepMCGenerator : public GeneratorBase {
    
public:

    /// Constructor(s).
    // No empty constructor. 'filename' *must* be specified for this generator to make sense.
    /* HepMCGenerator () {} */ 

    HepMCGenerator (const std::string& filename) { open(filename); }


    /// Destructor.
    ~HepMCGenerator () {}


    /// Method(s).
    virtual inline const arma::Mat<double>& next () {

        // Check whether generator is properly set up.
        check_();

        // Initialise generator input to zeros.
        m_data.zeros();

        // Fill histogram.
        TH2F hist ("tmp", "", shape()[0], -3.2, 3.2, shape()[1], -PI, PI);
        
        HepMC::GenEvent::particle_const_iterator p     = m_event->particles_begin();
        HepMC::GenEvent::particle_const_iterator p_end = m_event->particles_end();
        

        for ( ; p != p_end; p++) {
            // -- Keep only (calorimeter) visible final state particles.
            if ((*p)->status() != 1) { continue; }
            unsigned idAbs = abs((*p)->pdg_id());
            if (idAbs == 12 || idAbs == 14 || idAbs == 16 || idAbs == 13) { continue; }
        
            hist.Fill((*p)->momentum().eta(), (*p)->momentum().phi(), (*p)->momentum().perp() / 1000.);
        }

        // Fill 'm_data' matrix with content from 'hist'.
        HistFillMatrix(&hist, m_data);
        
        // Get next HepMC event.
        // This is done here, such that when HepMCGenerator::good is called, 
        // we're checking the *next* event, and therefore we don't risk 
        // generating a bad event.
        getNextHepMCEvent();
        
        return m_data;
    }

    virtual inline bool good () const { 
        
        if (!m_IO)    { INFO("Member object 'm_IO' is nullptr."); }
        if (!m_event) { INFO("Member object 'm_event' is nullptr."); }
        
        return m_IO && m_event; 
     }
    
    virtual inline bool open  (const std::string& filename) {         
        m_filename = filename;
        return open(); 
    }

    inline bool open  () { 

        // Check whether file name makes sense.
        if (m_filename == "") {
            ERROR("Filename not set. Exiting.");
            return false;
        }

        if (!fileExists(m_filename)) {
            ERROR("File '%s' does not exist. Exiting.", m_filename.c_str());
            return false;
        }

        // Get the HepMC IO objects from input stream of file, from file name.
        m_input = std::move(std::unique_ptr<std::istream>      (new std::fstream(m_filename.c_str(), std::ios::in)));
        m_IO    = std::move(std::unique_ptr<HepMC::IO_GenEvent>(new HepMC::IO_GenEvent(*m_input)));
        
        return getNextHepMCEvent(); 
    }

    // Read next event from HepMC file.
    inline bool getNextHepMCEvent () {
        
        if (m_IO)    { m_event = std::move(std::unique_ptr<HepMC::GenEvent>(m_IO->read_next_event())); } else {
            WARNING("Member object 'm_IO' is nullptr.");
            return false;
        }

        return true;
    }
    

private:

    /// Data member(s).
    // The name of the HepMC from which to generate the input.
    std::string m_filename = "";

    // Stream used to read the HepMC file.
    std::unique_ptr<std::istream> m_input = nullptr;

    // HepMC IO object to read HepMC events from file.
    std::unique_ptr<HepMC::IO_GenEvent> m_IO = nullptr;

    // HepMC event.
    std::unique_ptr<HepMC::GenEvent> m_event = nullptr;
    
};
#endif // USE_HEPMC

} // namespace

#endif // WAVENET_GENERATORS_H