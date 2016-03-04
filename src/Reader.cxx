#include "Reader.h"

void Reader::getNextHepMCEvent () {
    
    if (_event) { delete _event; _event = 0; }
    if (_IO)    { _event = _IO->read_next_event(); } else {
        cout << "In <Reader::getNextHepMCEvent>: ERROR: Member object '_IO' is null." << endl;
    }
    return;
}

bool Reader::open (const string& filename) {
    this->setFilename( filename );
    return this->open();
}

bool Reader::open () {
    if (_eventmode != EventMode::File) { return true; }
    
    if (_filename == "") {
        cout << "In <Reader::open>: ERROR: Filename not set. Exiting." << endl;
        return false;
    }
    if (!fileExists(_filename)) {
        cout << "In <Reader::open>: ERROR: File '" << _filename << "' does not exist. Exiting." << endl;
        return false;
    }
    
    _input = new std::fstream(_filename.c_str(), std::ios::in);
    _IO    = new HepMC::IO_GenEvent(*_input);

    this->getNextHepMCEvent();
    
    return true;
}

void Reader::close () {
    if (_IO    != NULL) { delete _IO;    _IO    = NULL; }
    if (_input != NULL) { delete _input; _input = NULL; }
    return;
}

void Reader::reset () {
    this->close();
    this->open();
    return;
}


Mat<double> Reader::next() {
    /* --- */
    //arma::arma_rng::set_seed(1);
    /* --- */
    switch (_eventmode) {
        case EventMode::File:
            return nextFile();
            break;
        case EventMode::Uniform:
            return nextUniform();
            break;
        case EventMode::Needle:
            return nextNeedle();
            break;
        case EventMode::Gaussian:
            return nextGaussian();
            break;
    }
    cout << "In <Reader::next>: EventMode was not set or recognised." << endl;
    return Mat<double>();
}


Mat<double> Reader::nextFile () {
    Mat<double> X;
    
    if (!good()) {
        cout << "In <Reader::next> ERROR: Reader not good. Exiting." << endl;
        return Mat<double> ();
    }
    TH2F hist ("tmp", "", _size, -3.2, 3.2, _size, -PI, PI);
    
    HepMC::GenEvent::particle_const_iterator p     = _event->particles_begin();
    HepMC::GenEvent::particle_const_iterator p_end = _event->particles_end();
    
    for ( ; p != p_end; p++) {
        // Keep only (calorimeter) visible final state particles.
        if ((*p)->status() != 1) { continue; }
        unsigned idAbs = abs((*p)->pdg_id());
        if (idAbs == 12 || idAbs == 14 || idAbs == 16 || idAbs == 13) { continue; }
    
        hist.Fill((*p)->momentum().eta(), (*p)->momentum().phi(), (*p)->momentum().perp() / 1000.);
        
    }

    X = HistToMatrix(hist);
    
    this->getNextHepMCEvent();
    
    return X;
    
}

Mat<double> Reader::nextUniform () {
    Mat<double> X (_size, _size, fill::randu);
    return X;
}

Mat<double> Reader::nextNeedle () {
    Mat<double> X (_size, _size, fill::randu);
    X.elem( find(X < 0.995) ) *= 0;
    return X;
}

Mat<double> Reader::nextGaussian () {
    
    Mat<double> shiftX (_size, _size, fill::zeros);
    Mat<double> shiftY (_size, _size, fill::zeros);
    
    int shiftIndX = int(2 * (arma::as_scalar(randu<mat>(1,1)) - 0.5) * (double)_size);
    int shiftIndY = int(2 * (arma::as_scalar(randu<mat>(1,1)) - 0.5) * (double)_size);
    
    shiftX.diag( shiftIndX ) += 1;
    shiftY.diag( shiftIndY ) += 1;
    
    if      (shiftIndX < 0) { shiftX.diag( _size + shiftIndX ) += 1; }
    else if (shiftIndX > 0) { shiftX.diag( _size - shiftIndX ) += 1; }
    if      (shiftIndY < 0) { shiftY.diag( _size + shiftIndY ) += 1; }
    else if (shiftIndY > 0) { shiftY.diag( _size - shiftIndY ) += 1; }
    
    // ... Gaussian blur
    
    Mat<double> h = linspace<mat>(-((double)_size-1)/2, ((double)_size-1)/2, _size);
    h = repmat(h, 1, _size);
    double sigma = 4;
    Mat<double> gauss = exp( - (square(h) + square(h.t())) / (2*sq(sigma)));
    gauss /= accu(gauss); // Unit integral.
    gauss *= max(max(gauss));
    
    // ...
    
    Mat<double> X = shiftX * gauss * shiftY;
    
    return X;
}

bool Reader::good () {

    if (_eventmode != EventMode::File) { return true; }
    
    if (!_IO) {
        cout << "In <Reader::good>: Member object '_IO' is null." << endl;
    }
    if (!_event) {
        cout << "In <Reader::good>: Member object '_event' is null." << endl;
    }
    
    return _IO && _event; //(_IO->rdstate() & std::ifstream::goodbit) ;
}


/* HepMC
 
 #include "HepMC/GenEvent.h"
 #include "HepMC/IO_GenEvent.h"
 
 [...]

 // The usual mess of reading from a HepMC file!
 std::istream* SignalFile = new std::fstream(SignalPath.c_str(), std::ios::in);
 HepMC::IO_GenEvent SignalIO(*SignalFile);
 HepMC::GenEvent* SignalEvent = SignalIO.read_next_event();

*/