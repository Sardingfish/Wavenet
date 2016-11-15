#include "Wavenet/GeneratorBase.h"

namespace wavenet {
    
bool GeneratorBase::open () {
    // Dummy implementation.
    return true; 
}

bool GeneratorBase::close () {
    // Dummy implementation.
    return true; 
}

bool GeneratorBase::setShape (const std::vector<unsigned>& shape) {

    // Initialiase variables.
    m_initialised = true;
    bool compress = false;

    // Check whether the requested dimensions are feasible.
    std::vector<unsigned> compressed;
    for (const unsigned& size : shape) {

        // If either dimension has size 1, we compress along that dimension. 
        // Keep all non-size-1 dimension for potential compression.
        compress |= (size == 1);
        if (size > 1) {
            compressed.push_back(size);
        }

        // If either dimension size is not radix 2, declare generator not 
        // properly initialised.
        m_initialised &= isRadix2(size);
    }

    
    if (!m_initialised) {

        // Produce warning if the dimension size were problematic.
        std::string shapeString = "";
        unsigned i = 0;
        for (const auto& size : shape) {
            if (i++ > 0) { shapeString += ", "; }
            shapeString += std::to_string(size);
        }
        WARNING("Requested shape {%s} is no good.", shapeString.c_str());
        m_shape = {};
        resize_();
        
    } else if (compress) {

        // Compress if needed.
        m_initialised = setShape(compressed);

    } else {

        // Otherwise, store accepted shape dimensions.
        m_shape = shape;
        if (m_shape.size() == 1) { m_shape.push_back(1); }
        
        resize_();

    } 

    return m_initialised;
}

bool GeneratorBase::resize_ () {

    // Check whether generator is properly initialised.
    if (!initialised()) {
        WARNING("Cannot resize generator which isn't properly initialised.")
        return false;
    }

    // Resize generator input object.
    m_data.resize(m_shape[0], m_shape[1]);

    return true;
}

bool GeneratorBase::check_ () {
    
    // Check whether generator is properly initialised.
    if (!initialised()) { 
        WARNING("Generator not properly initialised.");
        return false;
    }

    // Check whether generato r is in a good condition.
    if (!good()) {
        ERROR("Generator not good. Exiting.");
        return false;
    }

    return true;
}

} // namespace
