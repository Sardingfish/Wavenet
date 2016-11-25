#!/usr/bin/env bash

# Where to look for external installations.
EXTERNPATH=external

# Find paths to external installations.
INSTALLDIRS=`find $EXTERNPATH/* -name install`

# Add to LD_LIBRARY_PATH
for INSTALLDIR in ${INSTALLDIRS[@]}; do
    if [ -d $PWD/$INSTALLDIR/lib ]; then
	export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$PWD/$INSTALLDIR/lib
    fi
done
