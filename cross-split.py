#!/usr/bin/python

import sys
import numpy as np
import cPickle as pickle
from common import *

def make_indicators(values, order=None):
    if order is None:
        order = sorted(set(values))
    idcs = dict([ (v,i) for (i,v) in enumerate(order) ])
    indicators = np.zeros( (len(idcs), len(values)), dtype=np.float32 )
    for i, v in enumerate(values):
        indicators[idcs[v], i] = 1.
    return np.matrix(indicators)

def main():
    
    if len(sys.argv) != 5:
        sys.stderr.write('USAGE: cross-split.py [ratio] [input mfcc file] [training output file] [test output file]\n')
        sys.exit(1)

    ratio = float(sys.argv[1])

    in_file = open(sys.argv[2], 'rb') if sys.argv[2] != '-' else sys.stdin
    reader = MFCCReader(in_file)

    # ----------- #

    mels = []
    dcts = []
    wvls = []
    labels = []

    sys.stderr.write('loading data...\n')
    for packet in reader:
        if not isinstance(packet, FramePacket):
            continue
        mels.append(packet.mel_powers)
        dcts.append(packet.dct_coeffs)
        wvls.append(packet.wvl_coeffs)
        labels.append(packet.group_header.label)

    # ----------- #

    sys.stderr.write('computing frequencies...\n')

    n = len(labels)
    labelfreq = dict()
    for l in labels:
        labelfreq[l] = 1 + labelfreq.get(l, 0)

    sys.stderr.write('total samples: %d\nlabel frequencies: %r\n' % (n,labelfreq))

    # ----------- #

    minfreq = min(labelfreq.itervalues())
    sys.stderr.write('rarest label has frequency %d; filtering...\n' % minfreq)

    r = np.random.random(size = n)
    selector = [ r[i] <= 1. * minfreq / labelfreq[labels[i]] for i in xrange(n) ]

    mels   = [ x for (i,x) in enumerate(mels)   if selector[i] ]
    dcts   = [ x for (i,x) in enumerate(dcts)   if selector[i] ]
    wvls   = [ x for (i,x) in enumerate(wvls)   if selector[i] ]
    labels = [ x for (i,x) in enumerate(labels) if selector[i] ]
    
    # ----------- #

    sys.stderr.write('recomputing frequencies...\n')

    n = len(labels)
    labelfreq = dict()
    for l in labels:
        labelfreq[l] = 1 + labelfreq.get(l, 0)

    sys.stderr.write('remaining samples: %d\nadjusted frequencies: %r\n' % (n,labelfreq))

    # ----------- #

    sys.stderr.write('numpyfying...\n')
   
    lblnames = sorted(labelfreq.iterkeys())
    mels = np.matrix(mels).astype(np.float32).T
    dcts = np.matrix(dcts).astype(np.float32).T
    wvls = np.matrix(wvls).astype(np.float32).T
    labels = make_indicators(labels, order=lblnames)

    # ----------- #

    sys.stderr.write('splitting (ratio %f)...\n' % ratio)

    selector = np.random.random(size = n) <= ratio
    training_size = sum(selector)
    test_size = n - training_size
  
    training = dict(
        mels     = mels[: , selector],
        dcts     = dcts[: , selector],
        wvls     = wvls[: , selector],
        labels   = labels[: ,  selector],
        lblnames = lblnames
    )

    selector = ~selector
   
    test = dict(
        mels     = mels[: , selector],
        dcts     = dcts[: , selector],
        wvls     = wvls[: , selector],
        labels   = labels[: , selector],
        lblnames = lblnames
    )

    sys.stderr.write('training set: %d; test set: %d\n' % (training_size, test_size))

    # ----------- #

    sys.stderr.write('dumping...\n')

    with open(sys.argv[3], 'wb') as f:
        pickle.dump(training, f, -1)
    with open(sys.argv[4], 'wb') as f:
        pickle.dump(test, f, -1)

if __name__ == '__main__':
    main()
