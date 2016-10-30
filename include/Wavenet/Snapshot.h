#ifndef WAVELETML_SNAPSHOT_H
#define WAVELETML_SNAPSHOT_H

/**
 * @file Snapshot.h
 * @author Andreas Sogaard
 **/

// STL include(s).
#include <string>
#include <cstdio> /* snprintf */

// ROOT include(s).
// ...

// HepMC include(s).
// ...

// Armadillo include(s).
// ...

// WaveletML include(s).
#include "Wavenet/Utils.h"
#include "Wavenet/WaveletML.h"

class WaveletML;

using namespace std;

//template< class T>
class Snapshot {
    
public:
    
    // Constructor(s).
    Snapshot () {};
    Snapshot (const string& pattern) :
        _pattern(pattern)
    {};
    Snapshot (const int& number) :
        _number(number)
    {};
    Snapshot (const string& pattern, const int& number) :
        _pattern(pattern), _number(number)
    {};
    
    // Destructor(s).
    ~Snapshot () {};
    
    // High-level management method(s).
    void   next   ();
    bool   exists () { return fileExists(file()); }
    string file   ();
    
    void load (WaveletML* ML);
    void save (WaveletML* ML);
    
    inline int    number  () { return _number; }
    inline string pattern () { return _pattern; }
    
    Snapshot& operator++ ()    { ++_number; return *this; }
    Snapshot& operator-- ()    { --_number; return *this; }
    Snapshot  operator++ (int) { _number++; return *this; }
    Snapshot  operator-- (int) { _number--; return *this; }
    
    // Set method(s).
    void setPattern (const string& pattern) { _pattern = pattern; return; }
    void setNumber  (const int&    number)  { _number  = number;  return; }
    
private:
    
    string _pattern = "";
    int    _number  = 1;
    
};

#endif
