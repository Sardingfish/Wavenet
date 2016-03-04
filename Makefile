# Variables.
CC=clang++
INCPATH = ./include
SRCPATH = ./src
LIBPATH = ./lib
OUTPATH = ./output
ARMAPATH = /Users/A/Dropbox/PhD/Work/WaveletML/armadillo-6.500.4
HEPMCPATH = /Users/A/Dropbox/hep/HepMC-2.06.09/installation/
ROOTPATH = $(shell root-config --incdir)
INCFLAGS = -I$(INCPATH) -I$(ARMAPATH)/include -I$(HEPMCPATH)/include
ROOTCFLAGS = $(shell root-config --cflags)
ROOTLIBS   = -L$(ROOTSYS)/lib $(shell root-config --libs) -L$(HEPMCPATH)/lib

LINKFLAGS = -L$(LIBPATH) -L$(ARMAPATH) $(ROOTLIBS)
CXXFLAGS = --std=c++11 $(INCFLAGS) $(ROOTCFLAGS) 

# Rules.
.PHONY: all Run

libs: MatrixOperator LowpassOperator HighpassOperator WaveletML Reader Coach Snapshot Run Analyse Test

all: libs Run

MatrixOperator:
	@echo "[Compiling: $@]"
	$(CC) src/$@.cxx -o lib/$@.o -c -Wall -Werror -fpic -O2 $(CXXFLAGS)
	$(CC) -shared -o lib/lib$@.so lib/$@.o $(ROOTLIBS) $(LINKFLAGS) -larmadillo -DARMA_DONT_USE_WRAPPER -lblas -llapack

LowpassOperator:
	@echo "[Compiling: $@]"
	$(CC) src/$@.cxx -o lib/$@.o -c -Wall -Werror -fpic -O2 $(CXXFLAGS)
	$(CC) -shared -o lib/lib$@.so lib/$@.o $(ROOTLIBS) $(LINKFLAGS) -lMatrixOperator -larmadillo -DARMA_DONT_USE_WRAPPER -lblas -llapack


HighpassOperator:
	@echo "[Compiling: $@]"
	$(CC) src/$@.cxx -o lib/$@.o -c -Wall -Werror -fpic -O2 $(CXXFLAGS)
	$(CC) -shared -o lib/lib$@.so lib/$@.o $(ROOTLIBS) $(LINKFLAGS) -lMatrixOperator -larmadillo -DARMA_DONT_USE_WRAPPER -lblas -llapack


WaveletML:
	@echo "[Compiling: $@]"
	$(CC) src/$@.cxx -o lib/$@.o -c -Wall -Werror -fpic -O2 $(CXXFLAGS)
	$(CC) -shared -o lib/lib$@.so lib/$@.o $(ROOTLIBS) $(LINKFLAGS) -lSnapshot -lMatrixOperator  -lLowpassOperator -lHighpassOperator -larmadillo -DARMA_DONT_USE_WRAPPER -lblas -llapack

Reader:
	@echo "[Compiling: $@]"
	$(CC) src/$@.cxx -o lib/$@.o -c -Wall -Werror -fpic -O2 $(CXXFLAGS)
	$(CC) -shared -o lib/lib$@.so lib/$@.o $(ROOTLIBS) $(LINKFLAGS) -lHepMC -larmadillo -DARMA_DONT_USE_WRAPPER -lblas -llapack

Coach:
	@echo "[Compiling: $@]"
	$(CC) src/$@.cxx -o lib/$@.o -c -Wall -Werror -fpic -O2 $(CXXFLAGS)
	$(CC) -shared -o lib/lib$@.so lib/$@.o $(ROOTLIBS) $(LINKFLAGS) -lWaveletML -lReader -larmadillo -DARMA_DONT_USE_WRAPPER -lblas -llapack

Snapshot:
	@echo "[Compiling: $@]"
	$(CC) src/$@.cxx -o lib/$@.o -c -Wall -Werror -fpic -O2 $(CXXFLAGS)
	$(CC) -shared -o lib/lib$@.so lib/$@.o $(ROOTLIBS) $(LINKFLAGS) -larmadillo -DARMA_DONT_USE_WRAPPER -lblas -llapack

Run:
	@echo "[Compiling: $@]"
	$(CC) src/$@.cxx -o bin/$@.exe $(CXXFLAGS) $(LINKFLAGS) -lHepMC -lCoach -lReader -lWaveletML -lMatrixOperator -lLowpassOperator -lHighpassOperator -larmadillo -DARMA_DONT_USE_WRAPPER -lblas -llapack

Analyse:
	@echo "[Compiling: $@]"
	$(CC) src/$@.cxx -o bin/$@.exe $(CXXFLAGS) $(LINKFLAGS) -lHepMC -lSnapshot -lCoach -lReader -lWaveletML -lMatrixOperator -lLowpassOperator -lHighpassOperator -larmadillo -DARMA_DONT_USE_WRAPPER -lblas -llapack

Test:
	@echo "[Compiling: $@]"
	$(CC) src/$@.cxx -o bin/$@.exe $(CXXFLAGS) $(LINKFLAGS) -lWaveletML -lLowpassOperator -lMatrixOperator -larmadillo -DARMA_DONT_USE_WRAPPER -lblas -llapack

clean:
	@rm -f $(LIBPATH)/*.o
	@rm -f $(LIBPATH)/*.so
	@rm -f $(SRCPATH)/*~
	@rm -f $(INCPATH)/*~
	@rm -f $(OUTPATH)/output_*
	@rm -f ./*~

#g++ example1.cpp -o example1 -O2 -I/Users/A/Dropbox/PhD/Work/WaveletML/armadillo-6.500.4/include -DARMA_DONT_USE_WRAPPER -lblas -llapack