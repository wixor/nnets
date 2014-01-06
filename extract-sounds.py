#!./python

import sys, re
from common import *

def main():
    if len(sys.argv) != 5:
        sys.stderr.write('USAGE: extract-sounds.py [filename regexp] [label regexp] [input mfcc file] [output mfcc file]\n')
        sys.exit(1)

    filename_re = re.compile(sys.argv[1])
    label_re = re.compile(sys.argv[2])

    in_file = open(sys.argv[3], 'rb') if sys.argv[1] != '-' else sys.stdin
    reader = MFCCReader(in_file)

    out_file = open(sys.argv[4], 'wb') if sys.argv[2] != '-' else sys.stdout
    writer = MFCCWriter(out_file)

    accept = True
    for packet in reader:
        if isinstance(packet, GroupHeaderPacket):
            accept = filename_re.match(packet.filename) and \
                     label_re.match(packet.label)
            if accept:
                print 'matched filename %s, label %s' % (packet.filename, packet.label)
        if isinstance(packet, ProfilePacket) or accept:
            writer.write(packet)

if __name__ == '__main__':
    main()

