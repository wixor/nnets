#!./python

import sys, itertools, collections
import numpy as np
import cPickle as pickle
from common import *

class Dataset(object):
    __slots__ = ('mode', 'makeX', 'profile', 'n', 'X', 'Y', 'labelnames')

    def __init__(self, mode):
        self.mode = mode
        self.makeX = xmaker(mode)
        self.profile = None
        self.n = 0
        self.X = []
        self.Y = []

    def addFrame(self, frame):
        if is_clipped(frame):
            return
        self.profile = self.profile or frame.group_header.profile
        self.X.append(self.makeX(frame))
        self.Y.append(frame.group_header.label)
        self.n += 1

    def equalize(self):
        if 0 == self.n:
            return

        sys.stderr.write('equalizing...\n')

        labelfreq = collections.Counter(self.Y)
        minfreq = min(labelfreq.itervalues())

        sys.stderr.write('  total samples: %d\n' % self.n)
        sys.stderr.write('  label frequencies: %r\n' % labelfreq)
        sys.stderr.write('  rarest label has frequency %d\n' % minfreq)

        fraction = .5
        r = np.random.random(size = self.n)
        selector = [ r[i] <= fraction * minfreq / labelfreq[self.Y[i]] for i in xrange(self.n) ]

        self.X = [ x for (i,x) in enumerate(self.X) if selector[i] ]
        self.Y = [ y for (i,y) in enumerate(self.Y) if selector[i] ]
        self.n = len(self.X)

        sys.stderr.write('  adding silence\n')
        self.add_silence(int(fraction*minfreq))

        labelfreq = collections.Counter(self.Y)

        sys.stderr.write('  remaining samples: %d\n' % self.n)
        sys.stderr.write('  adjusted frequencies: %r\n' % labelfreq)

    def add_silence(self, rep):
        mel_filters = self.profile.mel_filters
        silence = self.profile.mel_power_threshold
        noise = 10.

        for i in xrange(rep):
            mel_powers = list(np.random.uniform(silence, silence+noise, mel_filters))

            packet = FramePacket(
                seq = 0, group_header = None, fft_powers = [], sample_offset = 0,
                mel_powers = mel_powers
            )

            self.X.append(self.makeX(packet))
            self.Y.append('sil')
            self.n += 1

    def numpyfy(self, labelnames):
        sys.stderr.write('numpyfying...\n')

        self.labelnames = labelnames
        labelnums = dict([ (x,i) for (i,x) in enumerate(labelnames) ])

        self.X = np.matrix(np.vstack(self.X).T)

        Y = np.zeros( (len(labelnames), self.n), dtype=np.float32 )
        for i, label in enumerate(self.Y):
            Y[labelnums[label], i] = 1.
        self.Y = np.matrix(Y)

    def shuffle(self):
        sys.stderr.write('shuffling...\n')
        perm = np.random.permutation(self.n)
        self.X = self.X[:, perm]
        self.Y = self.Y[:, perm]

    def dump(self, filename):
        sys.stderr.write('dumping to %s...\n' % filename)
        with open(filename, 'wb') as f:
            pickle.dump(dict(
                mode = self.mode,
                X = self.X,
                Y = self.Y,
                labelnames = self.labelnames),
                f, -1)

def main():
    if len(sys.argv) != 4:
        sys.stderr.write('USAGE: pre-nnet.py [mode] [output pickle file] [input mfcc file]\n')
        sys.exit(1)

    dataset = Dataset(sys.argv[1])

    with oread(sys.argv[3]) as in_file:
        for packet in MFCCReader(in_file):
            if isinstance(packet, FramePacket):
                dataset.addFrame(packet)

    dataset.equalize()

    labelnames = sorted(set(dataset.Y))

    dataset.numpyfy(labelnames)
    dataset.shuffle()
    dataset.dump(sys.argv[2])

if __name__ == '__main__':
    main()

