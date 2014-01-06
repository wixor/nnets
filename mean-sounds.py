#!./python

import sys, itertools
import numpy as np
from common import *

dbexp = 0.23025851
dblog = 4.3429448

class MeanBucket(object):
    __slots__ = ('mel_powers', 'fft_powers', 'dct_coeffs', 'wvl_coeffs', 'count', 'label')

    def __init__(self, label, frame):
        self.mel_powers = np.exp(np.array(frame.mel_powers)*dbexp)
        self.fft_powers = np.exp(np.array(frame.fft_powers)*dbexp)
        self.dct_coeffs = [0.] * len(frame.dct_coeffs)
        self.wvl_coeffs = [0.] * len(frame.wvl_coeffs)
        self.count = 1.
        self.label = label

    def add(self, frame):
        self.mel_powers += np.exp(np.array(frame.mel_powers)*dbexp)
        self.fft_powers += np.exp(np.array(frame.fft_powers)*dbexp)
        self.count += 1.
    
    def get_group_header(self):
        return GroupHeaderPacket(seq = None, profile = None,
            filename = '', label = self.label, sample_offset = 0)

    def get_frame(self):
        return FramePacket(seq = None, group_header = None,
            mel_powers = list(np.log(self.mel_powers / self.count)*dblog),
            fft_powers = list(np.log(self.fft_powers / self.count)*dblog),
            dct_coeffs = self.dct_coeffs,
            wvl_coeffs = self.wvl_coeffs,
            sample_offset = None)
def main():
    if len(sys.argv) != 3:
        sys.stderr.write('USAGE: mean-vowels.py [input mfcc file] [output mfcc file]\n')
        sys.exit(1)

    in_file = open(sys.argv[1], 'rb') if sys.argv[1] != '-' else sys.stdin
    reader = MFCCReader(in_file)

    out_file = open(sys.argv[2], 'wb') if sys.argv[2] != '-' else sys.stdout
    writer = MFCCWriter(out_file)

    buckets = dict()

    for packet in reader:
        if isinstance(packet, ProfilePacket):
            writer.write(packet)
            continue

        if not isinstance(packet, FramePacket):
            continue

        label = packet.group_header.label
        if label in buckets:
            buckets[label].add(packet)
        else:
            buckets[label] = MeanBucket(label, packet)

    for label in sorted(buckets.iterkeys()):
        bucket = buckets[label]
        writer.write(bucket.get_group_header())
        writer.write(bucket.get_frame())

if __name__ == '__main__':
    main()


