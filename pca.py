#!./python

import sys, itertools
import numpy as np
from common import *

def main():
    if len(sys.argv) != 2:
        sys.stderr.write('USAGE: pca.py [input mfcc file]\n')
        sys.exit(1)

    X = [ ]
    labels = { }

    print '# loading frames...'

    profile = None
    with oread(sys.argv[1]) as in_file:
        for packet in MFCCReader(in_file):
            if isinstance(packet, ProfilePacket):
                profile = packet
            elif isinstance(packet, FramePacket):
                if is_clipped(packet):
                    continue
                label = packet.group_header.label
                if label not in labels:
                    labels[label] = []
                labels[label].append(len(X))
                X.append(packet.mel_powers)
    print '# loaded %d frames' % len(X)

    X = np.array(X, dtype=np.float32).T
    
    subX = []
    s = min(map(len, labels.itervalues()))
    for idcs in labels.itervalues():
        idcs = np.random.choice(idcs, s, replace=False)
        subX.append( X[:, idcs] )
    X = np.matrix(np.hstack(subX))
    del subX
    del labels

    print '# got %d frames after sample count equalization; each label has %d' % (X.shape[1], s)
    
    print '# normalizing volume...'
    X -= np.average(X, axis=0)

    print '# centering...'
    X -= np.average(X, axis=1)

    print '# computing covariance...'
    m = X.shape[0] # number of dimentions
    n = X.shape[1] # number of samples
    M = (X * X.T) * (1. / n)

    print '# computing eigenvalues...'
    L, V = np.linalg.eigh(M)

    perm = np.argsort(L)[::-1]
    L = L[perm]
    V = V[:, perm]

    print '#\t' + '\t'.join(map(str,L))
    for i,f in enumerate(profile.mel_freqs[1:-1]):
        sys.stdout.write('%f\t' % f)
        for j in xrange(m):
            sys.stdout.write('%f\t' % V[i,j])
        print

    print
    print

    print '# applying volume normalization matrix...'
    V = (  np.matrix(np.identity(m, dtype=np.float32))
         - np.matrix(np.ones(m, dtype=np.float32)) * (1. / m)) * V

    print repr(V)
    
if __name__ == '__main__':
    main()


