#ifndef WAVELETML_COACH_H
#define WAVELETML_COACH_H

/**
 * @file Coach.h
 * @author Andreas Sogaard
 **/

// STL include(s).
#include <string>
#include <fstream>
#include <iostream>

// ROOT include(s).
// ...

// HepMC include(s).
// ...

// Armadillo include(s).
// ...

// WaveletML include(s).
#include "Wavenet/WaveletML.h"
#include "Wavenet/Reader.h"
#include "Wavenet/Utils.h"

using namespace std;

class Coach {
    
public:
    
    // Constructor(s).
    Coach (const string& name) :
        _name(name)
    {};
    
    // Destructor(s).
    ~Coach () {;};
    
    // Set method(s).
    inline void setName    (const string& name)      { _name    = name; return; }
    inline void setBasedir (const string& basedir);

    inline void setWaveletML (WaveletML* ML)         { _ML = ML; return; }
    inline void setReader    (Reader* reader)        { _reader = reader; return; }
    
    inline void setEvents  (const int&  events)      { setNevents(events); return; };
           void setNevents (const int& Nevents);

    inline void setEpochs  (const unsigned&  epochs) { setNepochs(epochs); return; };
    inline void setNepochs (const unsigned& Nepochs) { _Nepochs = Nepochs; return; };

    inline void setInits   (const unsigned&  inits)  { setNinits(inits); return; };
    inline void setNinits  (const unsigned& Ninits)  { _Ninits = Ninits; return; }
    
    inline void setCoeffs  (const unsigned&  coeffs) { setNcoeffs(coeffs); return; };
           void setNcoeffs (const unsigned& Ncoeffs);

    inline void setUseAdaptiveLearning  (const unsigned& useAdaptiveLearning) { _useAdaptiveLearning = useAdaptiveLearning; return; };
    
    inline void setPrintLevel  (const bool& printLevel) { _printLevel = printLevel; return; };

    
    // Get method(s).
    inline string getName () { return _name; }
    inline string    name () { return getName(); }
    
    inline string getBasedir () { return _basedir; }
    inline string    basedir () { return getBasedir(); }
    
    inline WaveletML* getWaveletML () { return _ML; }
    inline WaveletML*    waveletML () { return getWaveletML(); }

    inline Reader* getReader () { return _reader; }
    inline Reader*    reader () { return getReader(); }

    inline int    events  () { return getNevents(); }
    inline int    Nevents () { return getNevents(); }
    inline int getEvents  () { return getNevents(); }
    inline int getNevents () { return _Nevents; }
    
    inline unsigned    epochs  () { return getNepochs(); }
    inline unsigned    Nepochs () { return getNepochs(); }
    inline unsigned getEpochs  () { return getNepochs(); }
    inline unsigned getNepochs () { return _Nepochs; }
    
    inline unsigned    inits   () { return getNinits(); }
    inline unsigned    Ninits  () { return getNinits(); }
    inline unsigned getInits   () { return getNinits(); }
    inline unsigned getNinits  () { return _Ninits; }
    
    inline unsigned    coeffs  () { return getNcoeffs(); }
    inline unsigned    Ncoeffs () { return getNcoeffs(); }
    inline unsigned getCoeffs  () { return getNcoeffs(); }
    inline unsigned getNcoeffs () { return _Ncoeffs; }
    
    inline bool    useAdaptiveLearning () { return getUseAdaptiveLearning(); }
    inline bool getUseAdaptiveLearning () { return _useAdaptiveLearning; }
    
    inline unsigned printLevel () { return _printLevel; }
    
    // High-level management info.
    void run ();
    
private:
    
    string _name = "";
    string _basedir = "./output/";
    
    WaveletML* _ML    = NULL;
    Reader*    _reader = NULL;
    
    int      _Nevents = -1;
    unsigned _Nepochs = 1;
    unsigned _Ninits  = 1;
    unsigned _Ncoeffs = 2;
    bool     _useAdaptiveLearning = false;
    
    unsigned _printLevel = 2;
    
};

#endif