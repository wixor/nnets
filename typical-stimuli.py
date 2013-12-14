#!./python

import sys
import cPickle as pickle
import numpy as np
import scipy.fftpack
import wavelet

freqs = [0.0, 68.5, 143.7, 226.2, 316.8, 416.3, 525.5, 645.4, 777.0, 921.5, 1080.1, 1254.2, 1445.4, 1655.3, 1885.7, 2138.6, 2416.3, 2721.2, 3055.9, 3423.3, 3826.7, 4269.5, 4755.7, 5289.4, 5875.3, 6518.6, 7224.7, 8000.0]

def wavelet_matrix(n):
    M = []
    for i in xrange(n):
        A = [0.] * n
        A[i] = 1.
        wavelet.forward(A)
        M.append(A)
    return np.matrix(M, dtype=np.float32).T

def dct_matrix(n):
    M = []
    for i in xrange(n):
        A = [0.] * n
        A[i] = 1.
        M.append(scipy.fftpack.dct(A, type=2))
    return np.matrix(M, dtype=np.float32).T / (2.*n)

def main():
    if len(sys.argv) != 3:
        sys.stderr.write('USAGE: typical-stimuli.py [mels|dcts|wvls] [weights file]\n')
        sys.exit(1)

    with open(sys.argv[2], 'rb') as f:
        W = pickle.load(f)
        labelnames = W['labelnames']
        W = W['weights']

    outputs = len(labelnames)
    inputs = W.shape[0] / outputs - 1

    W = np.matrix(W.reshape((outputs,inputs+1)))

    mel_filters = len(freqs) - 2

    def pad(W):
        leftpad = np.zeros(( outputs, 1 ))
        rightpad = np.zeros(( outputs, mel_filters - inputs - 1 ))
        return np.matrix( np.hstack((leftpad, W[:, 1:], rightpad)) )

    if sys.argv[1] == 'mels':
        pass
    elif sys.argv[1] == 'dcts':
        W = pad(W) * dct_matrix(mel_filters)
    elif sys.argv[1] == 'wvls':
        W = pad(W) * wavelet_matrix(mel_filters)
    else:
        raise ValueError('unexpected mode: %s; must be dcts, wvls or mels' % sys.argv[1])

    for i,label in enumerate(labelnames):
        sys.stdout.write('"label %s" ' % label)
    print
    for j in xrange(mel_filters):
        sys.stdout.write('%f ' % freqs[j+1])
        for i in xrange(outputs):
            sys.stdout.write('%f ' % W[i,j])
        print

if __name__ == '__main__':
    main()
