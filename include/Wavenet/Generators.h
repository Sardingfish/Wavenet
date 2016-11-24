#ifndef WAVENET_GENERATORS_H
#define WAVENET_GENERATORS_H

/**
 * @file   Generators.h
 * @author Andreas Sogaard
 * @date   14 November 2016
 * @brief  Various derived generator classes.
 */

// STL include(s).
#include <string> /* std::string, std::stof */
#include <cmath> /* ceil, floor, log2, pow */
#include <algorithm> /* std::max, std::min */
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
#include "Wavenet/Utilities.h"
#include "Wavenet/GeneratorBase.h"


/**
 * @TODO: - (Optional) Change compilation structure, such that changes in header propagate directly
 */

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

    virtual inline bool good () { return true; }

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

    virtual inline bool good () { return true; }

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

    virtual inline bool good () { return true; }

    virtual inline bool open () { arma::arma_rng::set_seed_random(); return true; }
    
};


/**
 *  Derived class generating input from CSV files.
 */
class CSVGenerator : public GeneratorBase {

public:

    /// Constructor(s).
    // No empty constructor. 'filenames' *must* be specified for this generator to make sense.
    /* CSVGenerator () {} */

    CSVGenerator (const std::vector<std::string>& filenames) { open(filenames); }


    /// Destructor.
    ~CSVGenerator () {}


    /// Method(s).
    virtual inline const arma::Mat<double>& next () {

        // Check whether generator is properly set up.
        check_();

        // Initialise variably-sized vector holding the entries in each new line
        // read from input CSV file.
        std::vector<double> entries;

        // Initialise stream stream through which to read the entries
        std::vector<std::string> string_entries = split(m_line, ',');

        // Get number of entries.
        const unsigned N = string_entries.size();
        unsigned Nuse = N;

        // Determine number of entries to use.
        if (!isRadix2(N)) {
            if (m_usePadding) {
                Nuse = (unsigned) pow(2, ceil (log2(N)));
            } else {
                Nuse = (unsigned) pow(2, floor(log2(N)));
            }
        }

        // Dynamically set shape for each training example.
        m_shape = {Nuse, 1};

        // Initialise generator input right shape, filled with to zeros.
        m_data.resize(m_shape[0], m_shape[1]);
        m_data.zeros();

        // Initialise vector of entries to use.
        std::vector<std::string> string_entries_use (string_entries.begin(), string_entries.begin() + std::min(Nuse, N));

        // Fill data matrix from vector of (string) entries;
        for (unsigned i = 0; i < std::min(Nuse, N); i++) {
            m_data(i, 0) = std::stof(string_entries_use.at(i));
        }

        // Get next CSV event.
        // This is done here, such that when CSVGenerator::good is called, we're
        // checking the _next_ event, and therefore we don't risk generating bad
        // events.
        getNextCSV_();

        return m_data;
    }

    virtual inline bool good () {

        // Check whether input file stream is good.
        bool isGood = (m_input->good() && !m_input->eof());

        // If the event is bad, try to move to next file.
        return isGood || tryNextFile_();
     }

    inline bool open (const std::vector<std::string>& filenames) {
        m_filenames = filenames;
        m_current = 0;
        return open();
    }

    inline bool open () {

        // Manually set generator as initialised. This is usually done after
        // confirming that the requested data shape is valid, however, for
        // CSVGenerator no shape is specified, which means that the generator is
        // initialised by default (since the constructor requires file names to
        // be specified).
        m_initialised = true;

        // Check whether file name makes sense.
        if (m_filenames.size() == 0) {
            ERROR("Filenames not set. Exiting.");
            return false;
        }

        if (!fileExists(m_filenames[m_current])) {
            ERROR("File '%s' does not exist. Exiting.", m_filenames[m_current].c_str());
            return false;
        }

        // Open stream from current input file.
        m_input = std::move(std::unique_ptr<std::istream> (new std::fstream(m_filenames[m_current].c_str(), std::ios::in)));

        return getNextCSV_();
    }

    bool setShape (const std::vector<unsigned>& shape) {
        WARNING("This method has no effect for CSVGenerator, where the shape");
        WARNING(".. of the training data is determined from the input CSV");
        WARNING(".. alone. You can decide whether to pad or trim any data,");
        WARNING(".. which is not of length which is radix 2, by using the ");
        WARNING(".. 'setUsePadding(bool)' method.");
        return false;
    }

    inline void setUsePadding (const bool& usePadding) { m_usePadding = usePadding; }

    inline bool usePadding () { return m_usePadding; }


private:

    /// Internal method(s).
    // Read next event from CSV file.
    inline bool getNextCSV_ () {

        // Try to read the next line, separated by new-line tokens.
        if (!std::getline(*m_input, m_line, '\n')) { return false; }

        return true;
    }

    // Try to acces the next file. Return true if successful.
    inline bool tryNextFile_ () {

        // Check whether we're at the end of the list.
        bool endOfFileList = (m_current >= m_filenames.size() - 1);

        // Update index of current file to use.
        m_current = (++m_current % m_filenames.size());

        // Trigger a reset (new epoch) if at end of file list.
        if (endOfFileList) { return false; }

        // Otherwise try to move to the next file.
        return reset() && good();
    }


private:

    /// Data member(s).
    // The name of the CSV from which to generate the input.
    std::vector<std::string> m_filenames = {};

    // The index for the current file to use.
    unsigned m_current = 0;

    // Stream used to read the input CSV file.
    std::unique_ptr<std::istream> m_input = nullptr;

    // String holding the results from reading from the input CSV file.
    std::string m_line = "";

    // Whether to pad the data (if not radix 2) with zeros. Alternative is to
    // trim length to largest possible radix 2 number
    bool m_usePadding = false;

};


#ifdef USE_HEPMC
/**
 *  Derived class generating input from HepMC files. 
 */
class HepMCGenerator : public GeneratorBase {
    
public:

    /// Constructor(s).
    // No empty constructor. 'filenames' *must* be specified for this generator to make sense.
    /* HepMCGenerator () {} */ 

    HepMCGenerator (const std::vector<std::string>& filenames) { open(filenames); }

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
        // we're checking the _next_ event, and therefore we don't risk
        // generating bad events.
        getNextHepMCEvent_();
        
        return m_data;
    }

    virtual inline bool good () { 
        
        if (!m_IO)    { INFO("Member object 'm_IO' is nullptr."); }
        if (!m_event) { INFO("Member object 'm_event' is nullptr."); }
        
        bool isGood = (m_IO && m_event);

        // If the event is bad, try to move to next file.
        return isGood || tryNextFile_();
     }
    
    inline bool open (const std::vector<std::string>& filenames) {
        m_filenames = filenames;
        m_current = 0;
        return open(); 
    }

    inline bool open  () { 

        // Check whether file name makes sense.
        if (m_filenames.size() == 0) {
            ERROR("Filenames not set. Exiting.");
            return false;
        }

        if (!fileExists(m_filenames[m_current])) {
            ERROR("File '%s' does not exist. Exiting.", m_filenames[m_current].c_str());
            return false;
        }

        // Get the HepMC IO objects from input stream of file, from file name.
        m_input = std::move(std::unique_ptr<std::istream>      (new std::fstream(m_filenames[m_current].c_str(), std::ios::in)));
        m_IO    = std::move(std::unique_ptr<HepMC::IO_GenEvent>(new HepMC::IO_GenEvent(*m_input)));

        return getNextHepMCEvent_(); 
    }


private: 

    /// Internal method(s).
    // Read next event from HepMC file.
    inline bool getNextHepMCEvent_ () {
        
        if (m_IO) { m_event = std::move(std::unique_ptr<HepMC::GenEvent>(m_IO->read_next_event())); } else {
            WARNING("Member object 'm_IO' is nullptr.");
            return false;
        }

        return true;
    }

    // Try to acces the next file. Return true if successful.
    inline bool tryNextFile_ () {

        // Check whether we're at the end of the list.
        bool endOfFileList = (m_current >= m_filenames.size() - 1);

        // Update index of current file to use.
        m_current = (++m_current % m_filenames.size());

        // Trigger a reset (new epoch) if at end of file list.
        if (endOfFileList) { return false; }

        // Otherwise try to move to the next file.
        return reset() && good();
    }
    

private:

    /// Data member(s).
    // The names of the HepMC from which to generate the input.
    std::vector<std::string> m_filenames = {};

    // The index for the current file to use.
    unsigned m_current = 0;

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