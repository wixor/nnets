#!./python

import sys
import cPickle as pickle
import numpy as np
import scipy.fftpack
import wavelet

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
        sys.stderr.write('USAGE: mel-stimuli.py [input weights file] [output weights file]\n')
        sys.exit(1)

    with open(sys.argv[1], 'rb') as f:
        data = pickle.load(f)
    
    mel_filters = 17
    mode = data['mode']
    W = data['weights']
    outputs = len(data['labelnames'])

    W = np.matrix( W.reshape((outputs,8+1)) )

    bias = W[:, 0]
    W = W[:, 1:]

    if 'dcts' == mode:
        M = dct_matrix(mel_filters)[1:9, :]
    elif 'wvls' == mode:
        M = wavelet_matrix(mel_filters)[1:9, :]

    W = np.hstack(( bias, W*M ))

    data['mode'] = 'mels'
    data['weights'] = np.array(W).ravel()

    with open(sys.argv[2], 'wb') as f:
        pickle.dump(data, f, -1)

if __name__ == '__main__':
    main()
