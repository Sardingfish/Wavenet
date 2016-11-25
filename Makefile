# ----------------------------------------------------------------------------------
# External paths, to be set by user. Remember to also set the DYLD_LIBRARY_PATH. 
# If the packages were installed using the macros in the './scripts/' directory, 
# this environment variables can be set by calling '$ source scripts/setup.sh'.
ARMAPATH  =
HEPMCPATH =

# ----------------------------------------------------------------------------------
# Internal stuff. There should be no need to touch this.

# Get ROOT path from environment variable
ROOTPATH  := $(shell echo $$ROOTSYS)

# Variables
CXX = clang++
PACKAGENAME = Wavenet

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
PROGDIR = ./examples

# Extensions
SRCEXT = cxx
INCEXT = h

# Collections
SRCS := $(shell find $(SRCDIR) -name '*.$(SRCEXT)')
OBJS := $(patsubst $(SRCDIR)/%.$(SRCEXT),$(OBJDIR)/%.o,$(SRCS))
PROGSRCS := $(shell find $(PROGDIR) -name '*.$(SRCEXT)')
PROGS := $(patsubst $(PROGDIR)/%.$(SRCEXT),$(EXEDIR)/%.exe,$(PROGSRCS))
GARBAGE = $(OBJDIR)/*.o $(EXEDIR)/* $(LIBDIR)/*.so

# Dependencies
CXXFLAGS  = --std=c++11 -O3 -fPIC -funroll-loops -I$(INCDIR)
LINKFLAGS = -L$(LIBDIR)
LIBS      =

# -- Armadillo (necessary)
ifeq ($(strip $(ARMAPATH)),)
    $(info * --------------------------------------------------*)
    $(info | Path to Armadillo not provided. Please either:    |)
    $(info |  (1) set the ARMAPATH variable in the Makeile to  |)
    $(info |      the path of your existing local installation |)
    $(info |      or                                           |)
    $(info |  (2) run                                          |)
    $(info |       > source ./scripts/downloadArmadillo.sh     |)
    $(info |      to download and install Armadillo.           |)
    $(info * --------------------------------------------------*)
    $(error Exiting)
else
    # Checks to make sure that the provided path is sensible.
    ifeq ($(wildcard $(ARMAPATH)),)
        $(error Path pointed to by ARMAPATH ($(ARMAPATH)) does not exists)
    endif
    ifeq ($(wildcard $(ARMAPATH)/include),)
        $(error No 'include' directory was found in ARMAPATH ($(ARMAPATH)))
    endif
    ifeq ($(wildcard $(ARMAPATH)/lib),)
        $(error No 'lib' directory was found in ARMAPATH ($(ARMAPATH)))
    endif

    # If everything checks out, add flags.
    CXXFLAGS += -I$(ARMAPATH)/include -march=native -DARMA_DONT_USE_WRAPPER -DARMA_NO_DEBUG
    LINKFLAGS += -L$(ARMAPATH)/lib
    LIBS += -lArmadillo -lblas -llapack
endif

# -- HepMC (optional
ifeq ($(strip $(HEPMCPATH)),)
    $(info * --------------------------------------------------*)
    $(info | Path to HepMC not provided.                       |)
    $(info | As this is not necessary we will continue, but    |)
    $(info | some functionality will be disabled.              |)
    $(info | If you want to use HepMC, please either:          |)
    $(info |  (1) set the HEPMCPATH variable in Makefile to    |)
    $(info |      the path of your existing local installation |)
    $(info |      or                                           |)
    $(info |  (2) run                                          |)
    $(info |       > source ./scripts/downloadHepMC.sh         |)
    $(info |      to download and install HepMC.               |)
    $(info * --------------------------------------------------*)
else
    # Checks to make sure that the provided path is sensible.
    ifeq ($(wildcard $(HEPMCPATH)),)
        $(error Path pointed to by HEPMCPATH ($(HEPMCPATH)) does not exists)
    endif
    ifeq ($(wildcard $(HEPMCPATH)/include),)
        $(error No 'include' directory was found in HEPMCPATH ($(HEPMCPATH)))
    endif
    ifeq ($(wildcard $(HEPMCPATH)/lib),)
        $(error No 'lib' directory was found in HEPMCPATH ($(HEPMCPATH)))
    endif

    # If everything checks out, add flags.
    CXXFLAGS += -I$(HEPMCPATH)/include -DUSE_HEPMC
    LINKFLAGS += -L$(HEPMCPATH)/lib
    LIBS += -lHepMC
endif

# -- ROOT (optional)
ifeq ($(strip $(ROOTPATH)),)
    $(info * ------------------------------------------------- *)
    $(info | Path to ROOT not provided.                        |)
    $(info | As this is not necessary we will continue, but    |)
    $(info | some functionality will be disabled.              |)
    $(info * ------------------------------------------------- *)
else
    # Check to make sure that the provided paths is sensible?
    CXXFLAGS += $(ROOTCFLAGS) -DUSE_ROOT
    LINKFLAGS += -L$(ROOTSYS)/lib
    LIBS += $(ROOTLIBS)
endif


# Targets
all: $(PACKAGENAME) $(PROGS)

$(PACKAGENAME) : $(OBJS) 
	@mkdir -p $(LIBDIR)
	$(CXX) $(LINKFLAGS) -shared -O3 -o $(LIBDIR)/lib$@.so $(OBJS) $(LIBS)

$(OBJDIR)/%.o : $(SRCDIR)/%.$(SRCEXT) $(INCDIR)/$(PACKAGENAME)/%.$(INCEXT)
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(EXEDIR)/%.exe : $(PROGDIR)/%.$(SRCEXT)
	@mkdir -p $(EXEDIR)
	$(CXX) $< $(LINKFLAGS) -o $@ $(CXXFLAGS) $(LIBS) -l$(PACKAGENAME)

clean : 
	@rm -f $(GARBAGE)

