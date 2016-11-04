#!/usr/bin/env bash
export ARMAPATH=
export HEPMCPATH=

# --------------------------------------------------------------
# No need to touch this bit; it just ensures that you don't have
# to source this script everytime you run in a new shell.
sed -i.bak s,'\(^ARMAPATH *=\)\(.*\)$',"\1 ${ARMAPATH}",g Makefile
sed -i.bak s,'\(^HEPMCPATH *=\)\(.*\)$',"\1 ${HEPMCPATH}",g Makefile