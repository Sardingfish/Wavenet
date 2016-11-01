#include "Wavenet/GeneratorBase.h"

bool GeneratorBase::open () {
    return true; 
}

bool GeneratorBase::close () {
    return true; 
}

bool GeneratorBase::good () {
    return true; 
}

bool GeneratorBase::reset () {
    return close() && open(); 
}

bool GeneratorBase::setShape (const std::vector<unsigned int>& shape) {
    _initialised = true;
    bool compress = false;

    // Check whether the requested dimensions are feasible.
    std::vector<unsigned int> compressed;
    for (const unsigned int& size : shape) {
        // If either dimension has size 1, we compress along that dimension. Keep all non-size-1 dimension for potential compression.
        compress |= (size == 1);
        if (size > 1) {
            compressed.push_back(size);
        }
        // If either dimension size is not radix 2, declare generator not properly initialised.
        _initialised &= isRadix2(size);
    }

    
    if (!_initialised) {

        // Produce warning if the dimension size were problematic.
        std::cout << "<GeneratorBase::setShape> WARNING: Requested shape {"; 
        unsigned int i = 0;
        for (const auto& size : shape) {
            if (i++ > 0) {
                std::cout << ", ";
            }
            std::cout << size;
        }
        std::cout << "} is no good." << std::endl;
        _shape = {};
        _resize();
        
    } else if (compress) {

        // Compress if needed
        _initialised = setShape(compressed);

    } else {

        // Otherwise, store accepted shape dimensions.
        if (shape.size() == 1) { _shape = {shape[0], 1}; }

        _shape = shape;
        _resize();

    } 

    return _initialised;
}


std::vector<unsigned int> GeneratorBase::shape () { 
    return _shape; 
}

unsigned int GeneratorBase::dim () { 
    return _shape[1] == 1 ? 1 : 2; 
}

bool GeneratorBase::initialised () { 
    return _initialised; 
}

bool GeneratorBase::_resize () {
    _data.resize(_shape[0], _shape[1]);
    return true;
}