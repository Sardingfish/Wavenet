#ifndef WAVENET_GENERATORBASE_H
#define WAVENET_GENERATORBASE_H

/**
 * @file GeneratorBase.h
 * @author Andreas Sogaard
**/

// STL include(s).
#include <string> /* std::string */
#include <vector> /* std::vector */

// ROOT include(s).
// ...

// HepMC include(s).
// ...
 
// Armadillo include(s).
#include <armadillo>

// Wavenet include(s).
// ...


class GeneratorBase {

    /**
     * Abstract class for input generators. 
     *
     * Define derived classes to yield e.g. uniform, needle, gaussian, HepMC, or other custom input.
    **/

public:

    // Destructor.
    virtual ~GeneratorBase () {}

    // Method(s).
    virtual const arma::Mat<double>& next  () = 0;
    virtual bool open  () { return true; }
    virtual bool close () { return true; }
    virtual bool good  () { return true; }
    inline  bool reset () { return close() && open(); }

    inline bool setShape(const std::vector<unsigned int>& shape) {
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

    /*
    inline bool setShape (const unsigned int& first, const unsigned int& second) { 
        if (isRadix2(first) and isRadix2(second)) {
            _shape.first  = first;
            _shape.second = second;
        } else if (isRadix2(first) and second == 1) {
            _shape.first  = first;
            _shape.second = second;
        } else if (first == 1 and isRadix2(second)) {
            _shape.first  = second;
            _shape.second = first;
        } else {
            // ...
            return false;
        }
        return true; 
    }
    inline bool setShape (const arma::SizeMat& shape) { 
        return setShape(shape[0], shape[1]);
    }
    */
    inline std::vector<unsigned int> shape () { return _shape; }

    inline unsigned int dim () { return (_shape[1] == 1 ? 1 : 2); }
    inline bool initialised () { return _initialised; }


private:

    // Internal method(s).
    inline bool _resize () {
        _data.resize(_shape[0], _shape[1]);
        return true;
    }


protected:

    // Data member(s).
    bool _initialised = false;
    std::vector<unsigned int> _shape;
    arma::Mat<double> _data;

};

#endif