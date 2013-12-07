#!/usr/bin/python

import sys, itertools
import numpy as np
from common import *

def main():
    if len(sys.argv) != 2:
        sys.stderr.write('USAGE: coeff-histogram.py [input mfcc file]\n')
        sys.exit(1)

    in_file = open(sys.argv[1], 'rb') if sys.argv[1] != '-' else sys.stdin
    reader = MFCCReader(in_file)

    dct = dict()
    wvl = dict()

    for packet in reader:
        if not isinstance(packet, FramePacket):
            continue

        label = packet.group_header.label
        if not label in dct:
            dct[label] = [ [] for i in xrange(26) ]
        if not label in wvl:
            wvl[label] = [ [] for i in xrange(26) ]

        d = dct[label]
        w = wvl[label]

        for i,v in enumerate(packet.dct_coeffs):
            d[i].append(v)
        for i,v in enumerate(packet.wvl_coeffs):
            w[i].append(v)

    for label, clists in dct.iteritems():
        for coeffid, clist in enumerate(clists):
            histo, edges = np.histogram(clist, 100)

            print '%s-dct-%d' % (label,coeffid)
            for v,e1,e2 in itertools.izip(histo, edges, edges[1:]):
                print '%f %d' % (.5*(e1+e2), v)
            print '\n'
    
    for label, clists in wvl.iteritems():
        for coeffid, clist in enumerate(clists):
            histo, edges = np.histogram(clist, 100)

            print '%s-wvl-%d' % (label,coeffid)
            for v,e1,e2 in itertools.izip(histo, edges, edges[1:]):
                print '%f %d' % (.5*(e1+e2), v)
            print '\n'


if __name__ == '__main__':
    main()


