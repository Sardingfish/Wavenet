# Wavenet

__C++ package for learning optimal wavelet bases using a neural network approach.__

The functional basis is expressed in terms of a set of filter coefficients (known from wavelet analyses) and the learning is implemented using a neural network with gradient descent. The composite entity is called a _Wavenet_<sup>1</sup> below and is desribed in more detail in the companion note [1].

This package implements the wavenet transform using matrix algebra, for which we rely on the Armadillo C++ linear algebra  library [2]. If the HepMC package [3] for experimental particile physics event records is installed, a specialised Generator class, converting `HepMC::GenEvent`'s to Armadillo matrices, can be used. Finally, if the ROOT library [4] is installed, some of the bundled examples allow for producing output graphics showing the results of the learning process



## Installation

Generally, this package is developed for UNIX-based operating system. Therefore, getting it installed in  MacOS and Linux should be relatively smooth. Windows is not supported, but feel free to give it a try if you're feeling brave.

To install the base package itself, simply do:
```
$ git clone https://github.com/asogaard/Wavenet.git
$ cd Wavenet
```
That was easy. 

However, as mentioned above, the package depends on a few external packages that need to be installed and configured before you can
```
$ make
```
and run the _Wavenet_ package. How this is done is described below



### Dependencies

__LAPACK__, __BLAS__, and __Boost__ [5-7]. _Recommended._ Improves performance of matrix algebra in Armadillo (below). Pre-installed on MacOS systems. On Linux, install using APT as
```
$ sudo apt-get install liblapack-dev
$ sudo apt-get install libblas-dev
$ sudo apt-get install libboost-dev
```


__Armadillo__ [2]. _Necessary._ The _Wavenet_ package comes bundled with a utility script for downloading and installing Armadillo. Simply call
```
$ source scripts/downloadArmadillo.sh
```
This scripts installs the Armadillo package to `./external/` and updates the appropriate path (`ARMAPATH`) in the Makefile. See also the caveat below.

Alternatively, available on Linux using APT as 
```
$ sudo apt-get install libarmadillo-dev
```
on MacOS using [Homebrew](http://brew.sh/) as
```
$ brew install homebrew/science/armadillo
```
and through manual installation, cf. [the official webpage](http://arma.sourceforge.net/download.html). If installed in this way, the variable `ARMAPATH` at the top of the [Makefile](Makefile) needs to be updated to the installation directory and the `DYLD_LIBRARY_PATH` environment variable might need to be set manually, see below.


__ROOT__ [4]. _Recommended._ High energy physics framework. Used for plotting graphical results. Must be installed manually, following the instructions on [the official webpage](https://root.cern.ch/).


__HepMC__ [3]. _Optional._ Event record for high energy physics collision simulations. The _Wavenet_ package comes bundled with a utility script for downloading and installing HepMC. Simply call
```
$ source scripts/downloadHepMC.sh
```
This scripts installs the HepMC package to `./external/` and updates the appropriate path (`HEPMCPATH`) in the Makefile. See also the caveat below.

Alternatively, can be installed manually, cf. [the official webpage](http://hepmc.web.cern.ch/hepmc/). If installed in this way, the variable `HEPMCPATH` at the top of the [Makefile](Makefile) needs to be updated to the installation directory and the `DYLD_LIBRARY_PATH` environment variable might need to be set manually, see below.


### Caveat: `DYLD_LIBRARY_PATH`

If the external shared libraries, Armadillo and (optionally) HepMC, are not installed in standard locations (e.g. `/usr/local/`) the programs might be unable to find these by default. This is the case if the packages are installed using the bundled utility download scripts. In this case, you need to call
```
$ source scripts/setup.sh
```
which updates the environment variable `DYLD_LIBRARY_PATH` to point to the install directories in `./external/`, where the scripts install the external packages by default. This needs to be done in every new shell, so it might be smart to perform this call as part of the bash initialisation (i.e. put it in `~/.bash_profile` or similar). If you forget, the programs will let you know by throwing errors when you try to run.

If the packages are installed in another non-standard location, manually, make sure to set the appropriate environment variable(s) yourself.




## Structure

The central object in this package is the [Wavenet](include/Wavenet/Wavenet.h) class, which performs

1. forward transforms of input (1- or 2-dimensional) data into a set of neural network node activations or wavelet coefficients,
* inverse transforms or a set of wavelet coefficients into a "position-space" signal,
* backpropagation of sparisty errors on the wavelet coefficient, and
* learning updates based on the backpropagated errors from training examples.

The function for computing the sparsity- and regularisation costs, as well as the associated gradients, are located in [CostFunctions](include/Wavenet/CostFunctions.h).

The [LowpassOperator](include/Wavenet/LowpassOperator.h) and [HighpassOperator](include/Wavenet/HighpassOperator.h) classes, both deriving from the basic [MatrixOperator](include/Wavenet/MatrixOperator.h) class, are responsible for the implementation of the low- and high-pass filter operations in the _Wavenet_ transforms.

The [Coach](include/Wavenet/Coach.h) class manages the training<sup>1</sup> of Wavenet objects, possibly utilising more advanced learning methods such as adaptive learning rates and batch sizes as well as a variant of simulated annealing.

The training examples which go into the training are provided by a collection of [Generators](include/Wavenet/Generators.h), all deriving from the [GeneratorBase](include/Wavenet/GeneratorBase.h) class. By default, five generators come bundled with the package: `NeedleGenerator`, `UniformGenerator`, `GaussianGenerator`, `CSVGenerator`, and (optionally) `HepMCGenerator`. For specialised use, the user can added additional generator classes by following the provided examples.

The Wavenet objects can be save to, and loaded from, file using the [Snapshot](include/Wavenet/Snapshot.h) class, which also allows for easy iteration between save files from successive iterations, which the Coach class automatically takes care of.

The remaining files ([Logger](include/Wavenet/Logger.h), [Type](include/Wavenet/Type.h), and [Utilities](include/Wavenet/Utilities.h)) take care of pretty printing, type checking, and convenient utility functions.



## Example

Below is a minimal working example of the _Wavenet_ code in action:
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
This is the same code as in [examples/Example00.cxx](examples/Example00.cxx). For examples of graphical output generated with this package, please see [media](media).



## Licence

© Andreas Søgaard, 2016. Licensed under an [MIT License](LICENSE). If you use this package in your work (commerical, research, or otherwise), please give proper credit to the authors, e.g. through [1].



## References

[1] A. Søgaard, _Learning optimal wavelet bases using a neural network approach_, 2016. [arXiv:xxxx.yyyyy]  
[2] C. Sanderson and R. Curtin, _Armadillo: a template-based C++ library for linear algebra_, Journal of Open Source Software __1__ (2016) 26  
[3] M. Dobbs and J. B. Hansen, _The HepMC C++ Monte Carlo Event Record for High Energy Physics_, Comput. Phys. Commun. __134__ (2001) 41  
[4] R. Brun and F. Rademakers, _ROOT - An Object Oriented Data Analysis Framework_,
Proceedings AIHENP'96 Workshop, Lausanne, Sep. 1996, Nucl. Inst. & Meth. in Phys. Res. A __389__ (1997) 81-86. See also [http://root.cern.ch/](http://root.cern.ch/)  
[5] E. Anderson _et al._, _LAPACK Users' Guide_, 3rd ed., SIAM (1999). [[http://www.netlib.org/lapack](http://www.netlib.org/lapack)]  
[6] L. S. Blackford _et al._, _An Updated Set of Basic Linear Algebra Subprograms (BLAS)_, ACM Trans. Math. Soft., __28-2__ (2002) 135—151. [[http://www.netlib.org/blas](http://www.netlib.org/blas)]  
[7] _Boost C++ Libraries_, [http://www.boost.org](http://www.boost.org)


---

<sup>1</sup> Cute, right?
