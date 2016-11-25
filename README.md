# Wavenet

Package which enables the learning of maximally sparse orthonormal basis for arbitrary input 1- and 2D signal. The basis is expressed as a filter bank (known from wavelet analyses) and the learning is implemented using a neural network with gradient descent.

### Setup

...

#### Caveat 1: OS support

... Window, MacOS, Linux ...

#### Caveat 2: Linux

... BLAS, LAPACK ...

#### Caveat 3: DYLD_LIBRARY_PATH

... shared libraries, call `$ source scripts/setup.sh` to set environment variables to packages downloaded using the bundled scripts. Otherwise, make sure to do it manually, probably in `~/.bash_profile` or simiar.

### Example

...

Minimal working example:

```
#include "Wavenet/Generators.h" /* wavenet::NeedleGenerator */
#include "Wavenet/Wavenet.h" /* wavenet::Wavenet */
#include "Wavenet/Coach.h" /* wavevent::Coach */

int main (int argc, char* argv[]) {

    // Create Generator instance; here NeedleGenerator.
    wavenet::NeedleGenerator ng;
    ng.setShape({16,16});
    
    // Create Wavenet instance.
    wavenet::Wavenet wn;

    // Create Coach instance.
    wavenet::Coach coach ("Example00");
    coach.setGenerator(&ng);
    coach.setWavenet  (&wn);
    
    // Run the training.
    coach.run();

    return 1;
}
```

This is the same code as in [`examples/Example00.cxx`](examples/Example00.cxx).

### Structure

...


### Licence

...


### Contributions

...

