#!/usr/bin/env bash

# OS name.
UNAME=`uname`

# Directory in which to install HepMC.
EXTERNPATH="external"

# Name and version of library to install.
NAMEVERSION="HepMC-2.06.09"

# URL from which to download HepMC.
URL="http://lcgapp.cern.ch/project/simu/HepMC/download/${NAMEVERSION}.tar.gz"

if [ -d "${EXTERNPATH}/${NAMEVERSION}" ]; then 

    echo -e "\n\033[1m Target directory (${EXTERNPATH}/${NAMEVERSION}) already exists.\033[0m"

else

    echo -e "\n\033[1m Downloading ${NAMEVERSION} to '${EXTERNPATH}'.\033[0m"
    
    # Make sure that externals directory exists.
    mkdir -p $EXTERNPATH
    cd $EXTERNPATH

    # Download tarball (OS-specific).
    if [[ $UNAME == "Darwin"* ]]; then
        # -- MacOS
        curl -O $URL
    else
        # -- Linux
        wget $URL
    fi

    # Unzip and remove tarball.
    tar -xvzf ${NAMEVERSION}.tar.gz
    rm ${NAMEVERSION}.tar.gz

    # Make sure that build directory exist (should be separate from unzipped tarball contents).
    mkdir -p ${NAMEVERSION}-build/install
    cd ${NAMEVERSION}-build

    # Perform appropriate (c)make calls.
    cmake -DCMAKE_INSTALL_PREFIX=./install \
      -Dmomentum:STRING=GEV \
      -Dlength:STRING=MM \
      ../${NAMEVERSION}

    make
    make test
    make install

    # Move back to base directory.
    cd ../..

    # Update Makefile
    HEPMCPATH=${PWD}/${EXTERNPATH}/${NAMEVERSION}-build/install
    sed -i.bak s,'\(^HEPMCPATH *=\)\(.*\)$',"\1 ${HEPMCPATH}",g Makefile

    # Clean up.
    make clean

    # Done.
    echo -e "\033[1m Done.\033[0m"

fi # end: already installed

# PSA.
echo -e "\n\033[1m NOTE\033[0m: In order to let the compiler know the location of the shared libraries in"
echo -e " HepMC, the DYLD_LIBRARY_PATH must be set. This can be done by running:"
echo -e "   $ source scripts/setup.sh"
echo -e " Consider placing this line in your bash initialisation file(s), e.g."
echo -e " ~/.bash_profile or ~/.bashrc to automatically perform this setup for each new shell."
echo ""