#ifndef WAVELETML_Reader
#define WAVELETML_Reader

/**
 * @file Reader.h
 * @author Andreas Sogaard
 **/

// STL include(s).
#include <string>
#include <cmath> /* floor, round */
#include <fstream> /* std::ifstream::goodbit*/

// ROOT include(s).
// ...

// HepMC include(s).
#include "HepMC/GenEvent.h"
#include "HepMC/IO_GenEvent.h"

// Armadillo include(s).
#include <armadillo>

// WaveletML include(s).
#include "Wavenet/WaveletML.h"
#include "Wavenet/Utils.h"


using namespace std;
using namespace arma;


enum class EventMode
{
    File     = 0,
    Uniform  = 1,
    Needle   = 2,
    Gaussian = 3
};


//template< class T>
class Reader {
    
public:
    
    // Constructor(s).
    Reader () {;};
    
    // Destructor(s).
    ~Reader () {
        if (_event) { delete _event; _event = NULL; }
        if (_IO)    { delete _IO;    _IO    = NULL; }
        if (_input) { delete _input; _input = NULL; }
    }

    // Set method(s).
    void setSize      (const unsigned& size)       { _size      = size;      return; }
    void setFilename  (const string&   filename)   { _filename  = filename;  return; }
    void setEventMode (const EventMode& eventmode) { _eventmode = eventmode; return; }
    
    // High-level management method(s).
    bool        open (const string& filename);
    bool        open ();
    void        close ();
    void        reset ();
    Mat<double> next ();
    bool        good ();

    
protected:
    
    // Low-level management method(s).
    Mat<double> nextFile     ();
    Mat<double> nextUniform  ();
    Mat<double> nextNeedle   ();
    Mat<double> nextGaussian ();
    
    void getNextHepMCEvent ();
    
private:
    
    std::string _filename = "";
    
    std::istream*       _input = NULL;
    HepMC::IO_GenEvent* _IO    = NULL;
    HepMC::GenEvent*    _event = NULL;
    
    EventMode _eventmode = EventMode::File;
    
    // Placeholder member(s).
    unsigned _size = 64;
    
};

#endif