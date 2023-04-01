APPLNAME=BCD

GUROBI = /opt/gurobi751/linux64
GUROBIinc = $(GUROBI)/include
GUROBIlib = $(GUROBI)/lib

LAPACK = $(HOME)/lapack-3.7.1
LAPACKEinc = $(LAPACK)/LAPACKE/include
CBLASinc = $(LAPACK)/CBLAS/include

#F2C = $(LAPACKE)/F2CLIBS

APPLINCLS =	-I$(GUROBIinc) -I$(LAPACKEinc) -I$(CBLASinc)
APPLLIB = 	-L$(GUROBIlib) -L$(LAPACK)

APPLLIB += \
		-lpthread \
		-lm \
		-lgurobi_c++ \
		-lgurobi75 \
		-llapacke \
		-llapack \
		-lcblas \
		-lrefblas \
		-lgfortran

CC = g++

APPLLIB += -lstdc++ 


CXXLINKFLAGS =

DEBUG = -g -O0
#DEBUG= -O3 -DNDEBUG

#PROFILE=-pg
PROFILE=

CFLAGS = -Wall $(DEBUG) -c -DIL_STD $(APPLINCLS) $(PROFILE) -std=c++11 -mcmodel=medium
LDFLAGS = $(APPLLIB) $(CXXLINKFLAGS) $(PROFILE) -Wl,-rpath $(GUROBI)/lib

SOURCES=$(wildcard src/*.cpp)
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=BCD

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) 
	$(CC) -o $@ $(OBJECTS) $(LDFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	@rm -f $(EXECUTABLE) src/*.o
