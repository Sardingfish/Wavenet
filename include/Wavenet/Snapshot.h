#ifndef WAVENET_SNAPSHOT_H
#define WAVENET_SNAPSHOT_H

/**
 * @file Snapshot.h
 * @author Andreas Sogaard
 **/

// STL include(s).
#include <string> /* std::string */
#include <vector> /* std::vector */
#include <cstdio> /* snprintf */

// Wavenet include(s).
#include "Wavenet/Utilities.h"
#include "Wavenet/Logger.h"


namespace wavenet {

class Wavenet; /* To resolve circular dependence. */

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
    
    // Set method(s).
    inline void setPattern (const std::string& pattern) { _pattern = pattern; return; }
    inline void setNumber  (const int&         number)  { _number  = number;  return; }
    
    // Get method(s).
    inline int         number  () { return _number; }
    inline std::string pattern () { return _pattern; }
    
    // High-level management method(s).
           void        next   ();
    inline bool        exists () { return fileExists(file()); }
           std::string file   ();
    
    void load (Wavenet* wavenet);
    void save (Wavenet* wavenet);
    
    inline Snapshot& operator++ ()    { ++_number; return *this; }
    inline Snapshot& operator-- ()    { --_number; return *this; }
    inline Snapshot  operator++ (int) { _number++; return *this; }
    inline Snapshot  operator-- (int) { _number--; return *this; }
    
    
private:
    
    std::string _pattern = "";
    int         _number  = 0;
    
};

} // namespace

#endif // WAVENET_SNAPSHOT_H
