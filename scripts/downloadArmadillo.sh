#!/usr/bin/env bash

# OS name.
UNAME=`uname`

# Directory in which to install Armadillo.
EXTERNPATH="external"

# Name and version of library to install.
NAMEVERSION="armadillo-7.500.2"

# URL from which to download HepMC.
URL="http://sourceforge.net/projects/arma/files/${NAMEVERSION}.tar.xz"

if [ -d "${EXTERNPATH}/${NAMEVERSION}" ]; then 

    echo -e "\n\033[1m Target directory (${EXTERNPATH}/${NAMEVERSION}) already exists.\033[0m"

else

    echo -e "\n NOTE: Armadillo relies on other libraries (in particular LAPACK and BLAS) for"
    echo -e " performing the matrix algebra. These are automatically installed with MacOS,"
    echo -e " but Linux users might need to install them manually. See here for details:"
    echo -e "   http://arma.sourceforge.net/download.html"
    echo -e "\n\033[1m Downloading ${NAMEVERSION} to '${EXTERNPATH}'.\033[0m"
    
    # Make sure that externals directory exists.
    mkdir -p $EXTERNPATH
    cd $EXTERNPATH

    # Download tarball (OS-specific).
    if [[ $UNAME == "Darwin"* ]]; then
        # -- MacOS
        curl -OL $URL
    else
        # -- Linux
        wget $URL
    fi

    # Unzip and remove tarball.
    tar xf ${NAMEVERSION}.tar.xz
    rm ${NAMEVERSION}.tar.xz

    # Move to the unpacked directory.
    cd ${NAMEVERSION}

    # Create installation directory
    mkdir -p install

    # Make appropriate (c)make calls.
    cmake .
    make
    make install DESTDIR=install

    # Change directory structure.
    mv install/usr/local/* install/
    rmdir install/usr/local
    rmdir install/usr

    # Move back to base directory.
    cd ../..

    # Update Makefile
    ARMAPATH=${PWD}/${EXTERNPATH}/${NAMEVERSION}/install
    sed -i.bak s,'\(^ARMAPATH *=\)\(.*\)$',"\1 ${ARMAPATH}",g Makefile

    # Clean up.
    make clean

    # Done.
    echo -e "\033[1m Done.\033[0m"

fi # end: already installed

# PSA.
echo -e "\n\033[1m NOTE\033[0m: In order to let the compiler know the location of the shared libraries in"

echo -e " Armadillo, the DYLD_LIBRARY_PATH must be set. This can be done by running:"
echo -e "   $ source scripts/setup.sh"
echo -e " Consider placing this line in your bash initialisation file(s), e.g."
echo -e " ~/.bash_profile or ~/.bashrc to automatically perform this setup for each new shell."
echo ""