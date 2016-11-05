#ifndef WAVENET_COACH_H
#define WAVENET_COACH_H

/**
 * @file Coach.h
 * @author Andreas Sogaard
 **/

// STL include(s).
#include <iostream> /* std::cout */
#include <string> /* std::string */
#include <fstream> /* std::ofstream */

// Wavenet include(s).
#include "Wavenet/Utilities.h"
#include "Wavenet/Logger.h"
#include "Wavenet/Wavenet.h"
#include "Wavenet/GeneratorBase.h"


namespace wavenet {

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
    
           void setNumEvents (const int&      numEvents);
    inline void setNumEpochs (const unsigned& numEpochs) { _numEpochs = numEpochs; return; }
    inline void setNumInits  (const unsigned& numInits)  { _numInits  = numInits;  return; }
           void setNumCoeffs (const unsigned& numCoeffs);

    inline void setUseAdaptiveLearningRate (const unsigned& useAdaptiveLearningRate = true) { _useAdaptiveLearningRate = useAdaptiveLearningRate; return; }
    inline void setUseSimulatedAnnealing   (const unsigned& useSimulatedAnnealing   = true) { _useSimulatedAnnealing   = useSimulatedAnnealing;   return; }
           void setTargetPrecision         (const double&   targetPrecision);
    
    inline void setPrintLevel  (const bool& printLevel) { _printLevel = printLevel; return; }
    
    // Get method(s).
    inline std::string name ()    { return _name; }
    inline std::string basedir () { return _basedir; }
    inline std::string outdir ()  { return _basedir + _name + "/"; }
    
    inline Wavenet*       wavenet ()   { return _wavenet; }
    inline GeneratorBase* generator () { return _generator; }

    inline int          numEvents  () { return _numEvents; }
    inline unsigned int numEpochs  () { return _numEpochs; }
    inline unsigned int numInits   () { return _numInits; }
    inline unsigned int numCoeffs  () { return _numCoeffs; }
    
    inline bool   useAdaptiveLearningRate () { return _useAdaptiveLearningRate; }
    inline bool   useSimulatedAnnealing ()   { return _useSimulatedAnnealing; }
    inline double targetPrecision ()         { return _targetPrecision; }
    
    inline unsigned printLevel () { return _printLevel; }
    
    // High-level management info.
    bool run ();
    

private:
    
    std::string _name    = "";
    std::string _basedir = "./output/";
    
    Wavenet*       _wavenet   = nullptr;
    GeneratorBase* _generator = nullptr;
    
    int          _numEvents = 1000;
    unsigned int _numEpochs =    1;
    unsigned int _numInits  =    1;
    unsigned int _numCoeffs =    2;

    bool   _useAdaptiveLearningRate = false;
    bool   _useSimulatedAnnealing   = false;
    double _targetPrecision = -1;
    
    unsigned _printLevel = 3;
    
};

} // namespace

#endif // WAVENET_COACH_H
