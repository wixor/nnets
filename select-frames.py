#!./python

import sys, itertools
import numpy as np
from common import *

class Selector(object):
    __slots__ = ('makeX', 'group_header', 'frames')

    def __init__(self, makeX, group_header):
        self.makeX = makeX
        self.group_header = group_header
        self.frames = []

    def add(self, frame):
        if not is_clipped(frame):
            self.frames.append(frame)

    def flush(self, writer):
        if len(self.frames) < 5:
            sys.stderr.write('file %s, offset %d, label %s did not have 5 frames\n' % (
                 self.group_header.filename,
                 self.group_header.sample_offset,
                 self.group_header.label))
            return

        f1 = f2 = f3 = f4 = f5 = None
        s2 = s3 = s4 = s5 = 0
        
        best_i = None
        best_s = np.inf

        for i,f in enumerate(itertools.imap(self.makeX, self.frames)):
            f5,f4,f3,f2,f1 = f4,f3,f2,f1,f
            s5,s4,s3,s2 = s4,s3,s2,0

            if f2 is not None:
                d12 = f1 - f2
                d12 = np.sqrt(d12.dot(d12))
                s2 += d12
            if f3 is not None:
                d13 = f1 - f3
                d13 = np.sqrt(d13.dot(d13))
                s3 += d13
            if f4 is not None:
                d14 = f1 - f4
                d14 = np.sqrt(d14.dot(d14))
                s4 += d14
            if f5 is not None:
                d15 = f1 - f5
                d15 = np.sqrt(d15.dot(d15))
                s5 += d15

            if f5 is not None:
                s = s2 + s3 + s4 + s5
                if s < best_s:
                    best_s = s
                    best_i = i-4

        last_offset = -1000000000
        frame_spacing = self.group_header.profile.frame_spacing
        for f in self.frames[best_i:best_i+5]:
            if f.sample_offset != last_offset + frame_spacing:
                writer.write(self.group_header._replace(sample_offset = f.sample_offset))
            writer.write(f)
            last_offset = f.sample_offset
        
def main():
    if len(sys.argv) != 4:
        sys.stderr.write('USAGE: select-frames.py [mode] [input mfcc file] [output mfcc file]\n')
        sys.exit(1)

    makeX = xmaker(sys.argv[1])

    with oread(sys.argv[2]) as in_file:
        with owrite(sys.argv[3]) as out_file:

            writer = MFCCWriter(out_file)
            selector = None

            for packet in MFCCReader(in_file):
                if isinstance(packet, ProfilePacket):
                    writer.write(packet)
                    continue
                if not isinstance(packet, FramePacket):
                    continue

                if selector is None:
                    selector = Selector(makeX, packet.group_header)
                elif selector.group_header != packet.group_header:
                    selector.flush(writer)
                    selector = Selector(makeX, packet.group_header)

                selector.add(packet)

            if selector is not None:
                selector.flush(writer)

if __name__ == '__main__':
    main()

