#ifndef WAVENET_SNAPSHOT_H
#define WAVENET_SNAPSHOT_H

/**
 * @file Snapshot.h
 * @author Andreas Sogaard
 **/

// STL include(s).
#include <string> /* std::string */
#include <vector> /* std::vector*/
#include <cstdio> /* snprintf */

// Wavenet include(s).
#include "Wavenet/Utils.h"
#include "Wavenet/Logger.h"
#include "Wavenet/Wavenet.h"


class Wavenet;

class Snapshot : public Logger {
    
public:
    
    // Constructor(s).
    Snapshot () {};
    Snapshot (const std::string& pattern) :
        _pattern(pattern)
    {};
    Snapshot (const int& number) :
        _number(number)
    {};
    Snapshot (const std::string& pattern, const int& number) :
        _pattern(pattern), _number(number)
    {};
    
    // Destructor(s).
    ~Snapshot () {};
    
    // High-level management method(s).
    void        next   ();
    bool        exists () { return fileExists(file()); }
    std::string file   ();
    
    void load (Wavenet* ML);
    void save (Wavenet* ML);
    
    inline int         number  () { return _number; }
    inline std::string pattern () { return _pattern; }
    
    Snapshot& operator++ ()    { ++_number; return *this; }
    Snapshot& operator-- ()    { --_number; return *this; }
    Snapshot  operator++ (int) { _number++; return *this; }
    Snapshot  operator-- (int) { _number--; return *this; }
    
    // Set method(s).
    void setPattern (const std::string& pattern) { _pattern = pattern; return; }
    void setNumber  (const int&         number)  { _number  = number;  return; }
    
private:
    
    std::string _pattern = "";
    int         _number  = 1;
    
};

#endif
