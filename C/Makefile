EIGEN_PATH := C:/msys64/mingw64/include/eigen3

CC      := x86_64-w64-mingw32-gcc
CXX     := x86_64-w64-mingw32-g++

CFLAGS  := -Wall -g -O3 -march=native -ffast-math -funroll-loops -fPIC
CXXFLAGS := -Wall -g -O3 -std=c++17 -fPIC -I$(EIGEN_PATH) -fopenmp -Wno-class-memaccess
LDFLAGS := -shared -fopenmp

SRCS_C    := utils_backup.c
SRCS_CPP  := matrix_exp_eigen.cpp
OBJS      := $(SRCS_C:.c=.o) $(SRCS_CPP:.cpp=.o)
TARGET    := libhubbard.dll

.PHONY: all clean

all: $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

clean:
	rm -f $(OBJS) $(TARGET) __pycache__ *.pyc