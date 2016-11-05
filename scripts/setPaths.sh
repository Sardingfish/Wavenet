#!/usr/bin/env bash
export ARMAPATH=/Users/A/Dropbox/PhD/Work/WaveletML/armadillo-6.500.4
export HEPMCPATH=/Users/A/Dropbox/hep/HepMC-2.06.09/installation

# --------------------------------------------------------------
# No need to touch this bit; it just ensures that you don't have
# to source this script everytime you run in a new shell.
sed -i.bak s,'\(^ARMAPATH *=\)\(.*\)$',"\1 ${ARMAPATH}",g Makefile
sed -i.bak s,'\(^HEPMCPATH *=\)\(.*\)$',"\1 ${HEPMCPATH}",g Makefile