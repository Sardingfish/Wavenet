#ifndef WAVENET_LOGGER_H
#define WAVENET_LOGGER_H

/**
 * @file Logger.h
 * @author Andreas Sogaard
 **/

// STL include(s).
#include <iostream> /* std::cout, std::left */
#include <string> /* std::string */
#include <stdio.h> /* vprintf */
#include <stdarg.h> /* variadic functions */
#include <iomanip> /* std::setw */

// Armadillo include(s).
// ...

// Wavenet include(s).
#include "Wavenet/Type.h"

/* Macro only to be called from classes inheriting from Logger. */
#define ERROR(...)                  {this->_print(type(*this), __FUNCTION__, "ERROR",   __VA_ARGS__);}
#define WARNING(...)                {this->_print(type(*this), __FUNCTION__, "WARNING", __VA_ARGS__);}
#define INFO(...)                   {this->_print(type(*this), __FUNCTION__, "INFO",    __VA_ARGS__);}
#define DEBUG(...)   if (debug())   {this->_print(type(*this), __FUNCTION__, "DEBUG",   __VA_ARGS__);}
#define VERBOSE(...) if (verbose()) {this->_print(type(*this), __FUNCTION__, "VERBOSE", __VA_ARGS__);}
#define FCTINFO(...)                {Wavenet::Logger::_fctprint(__FUNCTION__, "INFO",    __VA_ARGS__);}


namespace Wavenet {

class Logger {
    
public:
    
    // Constructor(s).
    Logger () {};
    
    // Destructor.
    ~Logger () {};
    
    // Set method(s).
    inline void setDebug (const bool& debug = true) {
        _debug    = debug;
        _verbose &= debug;
        return;
    }
    
    inline void setVerbose (const bool& verbose = true) {
        _verbose  = verbose;
        _debug   |= verbose;
        return;
    }

    // Get method(s).
    inline bool debug   () { return _debug; }
    inline bool verbose () { return _verbose; }
    
    // Public print method(s).
    static void _fctprint (std::string fun, std::string type, std::string format, ...);

protected:
    
    // Internal method(s).
    void _print (std::string cls, std::string fun, std::string type, std::string format, ...);
    
    
protected:
    
    // Data member(s).
    bool _debug   = false;
    bool _verbose = false;
    
};

} // namespace

#endif // WAVENET_LOGGER_H
