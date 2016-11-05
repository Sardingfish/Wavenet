# ----------------------------------------------------------------------------------
# External paths, to be set by user.
ARMAPATH  =
HEPMCPATH =

# ----------------------------------------------------------------------------------
# Internal stuff. There should be no need to touch this.

# If external paths are not set already, try to get them from environment variables.
ifeq ($(ARMAPATH),)
    ARMAPATH  := $(shell echo $$ARMAPATH)
endif
ifeq ($(HEPMCPATH),)
    HEPMCPATH  := $(shell echo $$HEPMCPATH)
endif
ROOTPATH  := $(shell echo $$ROOTSYS)

# Variables.
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
LINKFLAGS = -O3 -L$(LIBDIR)

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
    ifeq ($(wildcard $(ARMAPATH)/libarmadillo*),)
        $(error No library of type '/libarmadillo*' was found in ARMAPATH ($(ARMAPATH)))
    endif

    # If everything checks out, add flags.
    CXXFLAGS += -I$(ARMAPATH)/include
    LINKFLAGS += -L$(ARMAPATH) -lArmadillo -DARMA_DONT_USE_WRAPPER -lblas -llapack
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
    LINKFLAGS += -L$(HEPMCPATH)/lib -lHepMC
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
    LINKFLAGS += -L$(ROOTSYS)/lib $(ROOTLIBS)
endif


# Targets
all: $(PACKAGENAME) $(PROGS)

$(PACKAGENAME) : $(OBJS) 
	@mkdir -p $(LIBDIR)
	$(CXX) -shared -O3 -o $(LIBDIR)/lib$@.so $(LINKFLAGS) $(OBJS)

$(OBJDIR)/%.o : $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(EXEDIR)/%.exe : $(PROGDIR)/%.$(SRCEXT)
	@mkdir -p $(EXEDIR)
	$(CXX) $< -o $@ $(CXXFLAGS) $(LINKFLAGS) -l$(PACKAGENAME)

clean : 
	@rm -f $(GARBAGE)

