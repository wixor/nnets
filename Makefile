CXX := g++
CXXFLAGS := -msse -msse2 -mfpmath=sse -march=native -ffast-math -O2 -Wall -Wshadow $(shell pkg-config --cflags fftw3 vorbisfile)
LDFLAGS := -lm $(shell pkg-config --libs fftw3 vorbisfile)

mfcc: mfcc.C

