CC := gcc
CFLAGS := -std=gnu99 -msse -msse2 -mfpmath=sse -march=native -ffast-math -O2 -Wall -Wshadow $(shell pkg-config --cflags fftw3)
LDFLAGS := -lm $(shell pkg-config --libs fftw3)

mfcc: mfcc.c

