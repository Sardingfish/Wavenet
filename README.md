# Wavenet

__C++ package for learning of optimal wavelet bases using a neural network approach.__

The basis is expressed as a filter bank, or a set of filter coefficients, (known from wavelet analyses) and the learning is implemented using a neural network with gradient descent. The compound entity is called a "Wavenet"<sup>1</sup> below and is desribed in more detail in the companion note [1].

This package implements the wavenet transform using matrix algebra, for which we rely on the Armadillo C++ linear algebra  library [2]. If the HepMC package [3] for experimental particile physics event records is installed, a specialised Generator class, convertinng `HepMC::GenEvent`'s to Armadillo matrices, can be used. Finally, if the ROOT library [4] is installed, some of the bundled examples allow for producing output graphics showing the results of the learning process

...


## Dependencies
---

* __LAPACK__, __BLAS__, and __Boost__ [5-7]. _Recommended._ Pre-installed on MacOS systems. On Linux, install using the APT as
```
# Linux
$ sudo apt-get install liblapack-dev
$ sudo apt-get install libblas-dev
$ sudo apt-get install libboost-dev
```
* __Armadillo__ [2]. _Necessary._ Available on Linux using APT as 
```
# Linux
$ sudo apt-get install libarmadillo-dev
```
on MacOS using [Homebrew](http://brew.sh/) as
```
# MacOS
$ brew install homebrew/science/armadillo
```
and through manual installation, cf. [the official webpage](http://arma.sourceforge.net/download.html). If installed in this way, the variable `ARMAPATH` at the top of the [Makefile](Makefile) needs to be updated to the installation directory. See also the caveat below.
* __ROOT__ [4]. _Recommended._ ...
* __HepMC__ [3]. _Optional._ Event record for simulated high energy physics. Can be installed manual, cf. [the official webpage](http://hepmc.web.cern.ch/hepmc/). Alternatively, the _Wavenet_ package comes bundled with a utility script for downloading and installing HepMC. Simply call
```
$ source scripts/downloadHepMC.sh
```
See also the caveat below.


### Caveat: `DYLD_LIBRARY_PATH`

If the external shared libraries Armadillo and (optionally) HepMC are not installed in standard locations (e.g. `/usr/local/`) the compiler might be unable to find these by default. This is the case if the packages are installed using the bundled utility download scripts. If this case, you need to call
```
$ source scripts/setup.sh
```
which updates the environment variable `DYLD_LIBRARY_PATH` to point to `./external/` where the scripts install the external packages by default. This needs to be done in every new shell, so it might be smart to perform this call as part of the bash initialisation (i.e. put it in `~/.bash_profile` or similar.)

If the packages are installed in another non-standard location, manually, make sure to set the appropriate environment variable(s) yourself.


## Installation
---

(_Note_: Please read this section in its entirety, including the caveats below, to avoid head aches when installing the package.)

To install the base package itself, simply do:
```
$ git clone https://github.com/asogaard/Wavenet.git
$ cd Wavenet
```

That was easy. However, since the package relies of external libraries for working, in particular Armadillo, some care has to be taken to interface these correctly with the package. There are generally two ways to do this:

1. __Manually__: If you already have Armadillo (and HepMC) installed, or want to do it manually, simply specify the path(s) to the installation directories in the [Makefile](Makefile). If these are not in standard locations (e.g. `/usr/local/`), see Caveat 3 below.
2. __Using installation scripts__: The package comes bundled with a few utilities scripts, located in [./scripts/](./scripts/), which should allow you to install Armadillo (and HepMC) with little hassle. Simply do:
```
$ source scripts/downloadArmadillo.sh
# and, optionally
$ source scripts/downloadHepMC.sh
```
This _should_ install both libraries in the [./externals/] directory and update the Makefile accordingly. See also Caveat 3 below.


#### Caveat 1: OS support

Generally, this package is developed for UNIX-based operating system. Therefore, getting it installed in  MacOS and Linux should be relatively smooth. Windows is supported, but feel free to give it a try if you're feeling brave.


#### Caveat 2: Linux

The 
... BLAS, LAPACK ...





## Structure
---

...



## Example
---

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

This is the same code as in [examples/Example00.cxx](examples/Example00.cxx).



## Licence
---
...


## Contributions
---

...


## References

[1] A. Søgaard, _Learning optimal wavelet bases using a neural network approach_, 2016. [arXiv:xxxx.yyyyy]  
[2] Armadillo, ...  
[3] M. Dobbs and J. Bech-Hansen, _...HepMC..._, 2006. [....]  
[4] ..., _...ROOT..._, yyyy. [....]  
[5] LAPACK
[6] BLAS
[7] Boost


---

<sup>1</sup> Cute, right?
