#ifndef WAVENET_TYPE_H
#define WAVENET_TYPE_H

#include <string>
#include <typeinfo>

/**
 * Taken from here: http://stackoverflow.com/a/4541470
 **/

std::string demangle(const char* name);

template <class T>
std::string type(const T& t) {
    return demangle(typeid(t).name());
}

#endif