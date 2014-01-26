#!./python 

import sys
import numpy as np
from common import *

def main():
    if len(sys.argv) != 2:
        sys.stderr.write('USAGE: project.py [input mfcc file]\n')
        sys.exit(1)

    makeX = xmaker('pca')

    X = []
    L = dict()

    with oread(sys.argv[1]) as in_file:
        for packet in MFCCReader(in_file):
            if isinstance(packet, FramePacket):
                label = packet.group_header.label
                if not label in L:
                    L[label] = []
                L[label].append(len(X))
                X.append(makeX(packet))

    X = np.vstack(X)

    #idcs = xrange(X.shape[0])
    #if True:
    s = min(map(len, L.itervalues()))
    for label, idcs in L.iteritems():
        idcs = np.random.choice(idcs, s, replace=False)
        print '"%s"' % label
        for i in idcs:
            print '%f %f %f' % (X[i,0],X[i,1],X[i,2])
        print
        print

if __name__ == '__main__':
    main()

