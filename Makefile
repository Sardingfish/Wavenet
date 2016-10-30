# Variables.
CXX = clang++
PACKAGENAME = WaveletML

# Root variables
ROOTCFLAGS := $(shell root-config --cflags)
ROOTLIBS   := $(shell root-config --libs)
ROOTGLIBS  := $(shell root-config --glibs)

# Directories
INCDIR = ./include
SRCDIR = ./src
OBJDIR = ./build
LIBDIR = ./lib
EXEDIR = ./bin
PROGDIR = ./Root

# Extensions
SRCEXT = cxx

# Collections
SRCS := $(shell find $(SRCDIR) -name '*.$(SRCEXT)')
OBJS := $(patsubst $(SRCDIR)/%.$(SRCEXT),$(OBJDIR)/%.o,$(SRCS))
PROGSRCS := $(shell find $(PROGDIR) -name '*.$(SRCEXT)')
PROGS := $(patsubst $(PROGDIR)/%.$(SRCEXT),$(EXEDIR)/%.exe,$(PROGSRCS))
GARBAGE = $(OBJDIR)/*.o $(EXEDIR)/* $(LIBDIR)/*.so

# Dependencies
CXXFLAGS  = --std=c++11 -O3 -fPIC -funroll-loops -I$(INCDIR) $(ROOTCFLAGS)
LINKFLAGS = -O3 -L$(LIBDIR) -L$(ROOTSYS)/lib $(ROOTLIBS) 

# Libraries
LIBS += $(ROOTLIBS)

# Externals
ARMAPATH = /Users/A/Dropbox/PhD/Work/WaveletML/armadillo-6.500.4
HEPMCPATH = /Users/A/Dropbox/hep/HepMC-2.06.09/installation/

CXXFLAGS += -I$(ARMAPATH)/include -I$(HEPMCPATH)/include
LINKFLAGS += -L$(HEPMCPATH)/lib -lHepMC -L$(ARMAPATH) -lArmadillo -DARMA_DONT_USE_WRAPPER -lblas -llapack

# Targets
all: $(PACKAGENAME) $(PROGS)

$(PACKAGENAME) : $(OBJS) 
	$(CXX) -shared -O3 -o $(LIBDIR)/lib$@.so $(LINKFLAGS) $(OBJS) $(LIBS) 

$(OBJDIR)/%.o : $(SRCDIR)/%.$(SRCEXT)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(EXEDIR)/%.exe : $(PROGDIR)/%.$(SRCEXT)
	$(CXX) $< -o $@ $(CXXFLAGS) $(LINKFLAGS) -l$(PACKAGENAME)
 
clean : 
	@rm -f $(GARBAGE)

