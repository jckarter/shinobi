#Source code directrory
SRCDIR    := src
INTERMDIR := obj

# Name of the execuatable to build
TARGET    := shinobi

# Cuda source files (compiled with nvcc)
CUFILES   := $(SRCDIR)/Shinobi.cu \
$(SRCDIR)/Utils/Sort.cu \
$(SRCDIR)/Utils/CUDAUtil.cu

MACHINE   := $(shell uname -s)
#Additional libraries
LIBS      := -L/opt/local/lib -L/usr/local/cuda/lib -lpng -lcudart -lcutil -framework OpenGL -lGLEW -lSDLmain -lSDL

# C/C++ source files (compiled with gcc / c++)
CCFILES := $(shell find $(SRCDIR) -iname '*.cpp')

################################################################################
# Rules and targets

.SUFFIXES : .cu 

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# Compilers
CUDA_INSTALL_PATH := /usr/local/cuda
CUDA_SDK_DIR := /Developer/CUDA
NVCC       := $(CUDA_INSTALL_PATH)/bin/nvcc
CXX        := g++
CC         := gcc
LINK       := g++ -fPIC

# Includes
INCLUDES  += -I$(SRCDIR) -I/opt/local/include -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SDK_DIR)/common/inc -I/opt/local/include/SDL
CXXFLAGS += -arch i386 $(INCLUDES)
CFLAGS += -arch i386 $(INCLUDES)
NVCCFLAGS += $(INCLUDES)


ifeq ($(sm_13), 1)
	NVCCFLAGS   += -arch sm_13
else ifeq ($(sm_12), 1)
	NVCCFLAGS   += -arch sm_12
else ifeq ($(sm_11), 1)
	NVCCFLAGS   += -arch sm_11
else
	NVCCFLAGS   += -arch sm_10
endif

ifeq ($(dbg),1)
        COMMONFLAGS += -g
        NVCCFLAGS   += -D_DEBUG
else
        COMMONFLAGS += -O2
        NVCCFLAGS   += -DNDEBUG --compiler-options -fno-strict-aliasing -use_fast_math -maxrregcount=32
endif

NVCCFLAGS  += -ccbin ccbin $(COMMONFLAGS)
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)

COMPILATIONPHASE := 

OBJS :=

ifeq ($(cubin), 1)
	COMPILATIONPHASE += -cubin
	OBJS += $(CUFILES:$(SRCDIR)/%.cu=./$(SRCDIR)/%.cubin)
else
	COMPILATIONPHASE += -c
	OBJS += $(CCFILES:$(SRCDIR)/%.cpp=./$(SRCDIR)/%.o)
	OBJS += $(CUFILES:$(SRCDIR)/%.cu=./$(SRCDIR)/%.o)
endif



LDFLAGS += -arch i386 $(LIBS) -L$(CUDA_INSTALL_PATH)/lib -L$(CUDA_SDK_DIR)/lib -L$(CUDA_SDK_DIR)/common/lib/$(OSLOWER) 



all: ccbin $(TARGET)

$(SRCDIR)/%.cubin : $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $(COMPILATIONPHASE) $<
	
$(SRCDIR)/%.o : $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $(COMPILATIONPHASE) $<

$(SRCDIR)/%.o : $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS)  -c $< -o $@

$(TARGET): $(OBJS)
	$(LINK) -o $(TARGET) $(OBJS) $(LDFLAGS)

ccbin :
	mkdir ccbin
	ln -sf /usr/bin/g++ ccbin/g++
	ln -sf /usr/bin/gcc ccbin/gcc
	
clean :
	rm -f $(OBJS)
	rm -f $(TARGET)
	rm -rf ccbin
