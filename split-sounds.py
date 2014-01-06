#!./python

import sys, re
from common import *

def main():
    if len(sys.argv) != 3:
        sys.stderr.write('USAGE: split-sounds.py [input mfcc file] [output mfcc prefix]\n')
        sys.exit(1)

    in_file = open(sys.argv[1], 'rb') if sys.argv[1] != '-' else sys.stdin
    reader = MFCCReader(in_file)

    outs = dict()

    last_group_header = None

    for packet in reader:
        if not isinstance(packet, FramePacket):
            continue

        label = packet.group_header.label

        if not label in outs:
            out_file = open(sys.argv[2] + '-' + label + '.mfcc', 'wb')
            writer = MFCCWriter(out_file)
            writer.write(packet.group_header.profile)
            outs[label] = (out_file, writer)
        else:
            writer = outs[label][1]

        if last_group_header != packet.group_header:
            writer.write(packet.group_header)
            last_group_header = packet.group_header

        writer.write(packet)

if __name__ == '__main__':
    main()

