#!./python

import numpy as np

class Wrapper(object):
    __slots__ = ('_src', '_n')
    def __init__(self, src, n):
        self._src = src
        self._n = n
    def __len__(self):
        return self._n
    def _wrap(self, i):
        while i < 0 or i >= self._n:
            if i < 0: i = -i
            if i >= self._n: i = (self._n-1) - (i - (self._n-1))
        return i
    def __getitem__(self, i):
        return self._src[self._wrap(i)]
    def __setitem__(self, i, v):
        self._src[self._wrap(i)] = v

def predict(T, i):
    #return (9. * (T[i+1] + T[i-1]) - (T[i+3] + T[i-3])) * 0.0625
    return .5 * (T[i-1] + T[i+1])
def update(T, i):
    #return (9. * (T[i+1] + T[i-1]) - (T[i+3] + T[i-3])) * 0.03125
    return .25 * (T[i-1] + T[i+1])

def forward_step(T, n):
    Twrap = Wrapper(T, n)
    for i in xrange(1, n, 2):
        Twrap[i] -= predict(Twrap, i)
    for i in xrange(0, n, 2):
        Twrap[i] += update(Twrap, i)

    T[:n] = [ T[2*i]   for i in xrange((n+1)/2) ] + \
            [ T[2*i+1] for i in xrange(n/2) ]

def backward_step(T, n):
    T1 = [ T[i]           for i in xrange((n+1)/2) ]
    T2 = [ T[i + (n+1)/2] for i in xrange(n/2) ]
    T[0:n:2] = T1
    T[1:n:2] = T2
    
    Twrap = Wrapper(T, n)
    for i in xrange(0, n, 2):
        Twrap[i] -= update(Twrap, i)
    for i in xrange(1, n, 2):
        Twrap[i] += predict(Twrap, i)

def forward(T):
    def f(T, n):
        if n >= 2:
            forward_step(T, n)
            f(T, (n+1)/2)
    f(T, len(T))

def backward(T):
    def f(T, n):
        if n >= 2:
            f(T, (n+1)/2)
            backward_step(T, n)
    f(T, len(T))
