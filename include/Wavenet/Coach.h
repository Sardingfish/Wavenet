#ifndef WAVENET_COACH_H
#define WAVENET_COACH_H

/**
 * @file Coach.h
 * @author Andreas Sogaard
 **/

// STL include(s).
#include <string> /* std::string */
#include <fstream>
#include <iostream>

// Armadillo include(s).
// ...

// Wavenet include(s).
#include "Wavenet/Utils.h"
#include "Wavenet/Logger.h"
#include "Wavenet/Wavenet.h"
#include "Wavenet/GeneratorBase.h"


class Coach : Logger  {
    
public:
    
    // Constructor(s).
    Coach (const std::string& name) :
        _name(name)
    {};
    
    // Destructor(s).
    ~Coach () {;};
    
    // Set method(s).
    inline void setName    (const std::string& name) { _name    = name; return; }
    inline void setBasedir (const std::string& basedir);

    inline void setWavenet   (Wavenet* wavenet)          { _wavenet   = wavenet;   return; }
    inline void setGenerator (GeneratorBase*  generator) { _generator = generator; return; }
    
           void setNevents (const int&      Nevents);
    inline void setNepochs (const unsigned& Nepochs) { _Nepochs = Nepochs; return; };
    inline void setNinits  (const unsigned& Ninits)  { _Ninits  = Ninits;  return; }
           void setNcoeffs (const unsigned& Ncoeffs);

    inline void setUseAdaptiveLearning  (const unsigned& useAdaptiveLearning) { _useAdaptiveLearning = useAdaptiveLearning; return; };
    inline void setUseAdaGrad           (const unsigned& useAdaGrad)          { _useAdaGrad          = useAdaGrad;          return; };
    
    inline void setPrintLevel  (const bool& printLevel) { _printLevel = printLevel; return; };

    
    // Get method(s).
    inline std::string name ()    { return _name; }
    inline std::string basedir () { return _basedir; }
    
    inline Wavenet*       wavenet ()   { return _wavenet; }
    inline GeneratorBase* generator () { return _generator; }

    inline int          events  () { return _Nevents; }
    inline unsigned int epochs  () { return _Nepochs; }
    inline unsigned int inits   () { return _Ninits; }
    inline unsigned int coeffs  () { return _Ncoeffs; }
    
    inline bool useAdaptiveLearning () { return _useAdaptiveLearning; }
    inline bool useAdaGrad ()          { return _useAdaGrad; }
    
    inline unsigned printLevel () { return _printLevel; }
    
    // High-level management info.
    void run ();
    
private:
    
    std::string _name    = "";
    std::string _basedir = "./output/";
    
    Wavenet*       _wavenet   = nullptr;
    GeneratorBase* _generator = nullptr;
    
    int          _Nevents = -1;
    unsigned int _Nepochs =  1;
    unsigned int _Ninits  =  1;
    unsigned int _Ncoeffs =  2;

    bool _useAdaptiveLearning = false;
    bool _useAdaGrad          = false;
    
    unsigned _printLevel = 2;
    
};

#endif