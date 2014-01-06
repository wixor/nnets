#!./python

import sys
import numpy as np
import cPickle as pickle
from common import *

class Dataset(object):
    __slots__ = ('name',  'mels', 'dcts', 'wvls', 'labels', 'labelnames', 'n')

    def __init__(self, name):
        self.name = name
        self.mels = []
        self.dcts = []
        self.wvls = []
        self.labels = []
        self.n = 0

    def addFrame(self, frame):
        self.mels.append(frame.mel_powers)
        self.dcts.append(frame.dct_coeffs)
        self.wvls.append(frame.wvl_coeffs)
        self.labels.append(frame.group_header.label)
        self.n += 1

    def equalize(self):
        if 0 == self.n:
            return

        sys.stderr.write('[%s] equalizing...\n' % self.name)

        labelfreq = dict()
        for l in self.labels:
            labelfreq[l] = 1 + labelfreq.get(l, 0)
        minfreq = min(labelfreq.itervalues())

        sys.stderr.write('  total samples: %d\n' % self.n)
        sys.stderr.write('  label frequencies: %r\n' % labelfreq)
        sys.stderr.write('  rarest label has frequency %d\n' % minfreq)

        r = np.random.random(size = self.n)
        selector = [ r[i] <= 1. * minfreq / labelfreq[self.labels[i]] for i in xrange(self.n) ]

        self.mels   = [ x for (i,x) in enumerate(self.mels)   if selector[i] ]
        self.dcts   = [ x for (i,x) in enumerate(self.dcts)   if selector[i] ]
        self.wvls   = [ x for (i,x) in enumerate(self.wvls)   if selector[i] ]
        self.labels = [ x for (i,x) in enumerate(self.labels) if selector[i] ]
        self.n = len(self.labels)
        
        labelfreq = dict()
        for l in self.labels:
            labelfreq[l] = 1 + labelfreq.get(l, 0)

        sys.stderr.write('  remaining samples: %d\n' % self.n)
        sys.stderr.write('  adjusted frequencies: %r\n' % labelfreq)

    def numpyfy(self, labelnames):
        sys.stderr.write('[%s] numpyfying...\n' % self.name)
        self.mels = np.matrix(self.mels).astype(np.float32).T
        self.dcts = np.matrix(self.dcts).astype(np.float32).T
        self.wvls = np.matrix(self.wvls).astype(np.float32).T

        self.labelnames = labelnames
        labelnums = dict([ (x,i) for (i,x) in enumerate(labelnames) ])

        lmatrix = np.matrix(np.zeros( (len(labelnames), len(self.labels)), dtype=np.float32 ))
        for i, v in enumerate(self.labels):
            lmatrix[labelnums[v], i] = 1.
        self.labels = lmatrix

    def dump(self, filename):
        sys.stderr.write('[%s] dumping to %s...\n' % (self.name, filename))
        with open(filename, 'wb') as f:
            pickle.dump(dict(
                name=self.name,
                mels=self.mels,
                dcts=self.dcts,
                wvls=self.wvls,
                labels=self.labels,
                labelnames=self.labelnames),
                f, -1)

def main():
    if len(sys.argv) != 4:
        sys.stderr.write('USAGE: cross-split.py [input mfcc file] [training output file] [test output file]\n')
        sys.exit(1)

    in_file = open(sys.argv[1], 'rb') if sys.argv[1] != '-' else sys.stdin
    reader = MFCCReader(in_file)

    training_set = Dataset('training')
    test_set = Dataset('test')

    test_people = ('ao1m1', 'pw1m1', 'mr1m1', 'ps1m1', 'sw1m1')

    sys.stderr.write('loading data...\n')
    for packet in reader:
        if not isinstance(packet, FramePacket):
            continue
        filename = packet.group_header.filename
        test = any([ filename.startswith(x) for x in test_people ])
        if test:
            test_set.addFrame(packet)
        else:
            training_set.addFrame(packet)

    training_set.equalize()
    test_set.equalize()

    labelnames = sorted(set(training_set.labels) | set(test_set.labels))

    training_set.numpyfy(labelnames)
    test_set.numpyfy(labelnames)

    training_set.dump(sys.argv[2])
    test_set.dump(sys.argv[3])

if __name__ == '__main__':
    main()
