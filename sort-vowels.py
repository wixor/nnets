#!./python

import sys, itertools
from common import *

def main():
    if len(sys.argv) != 3:
        sys.stderr.write('USAGE: sort-vowels.py [input mfcc file] [output mfcc file]\n')
        sys.exit(1)

    in_file = open(sys.argv[1], 'rb') if sys.argv[1] != '-' else sys.stdin
    reader = MFCCReader(in_file)

    out_file = open(sys.argv[2], 'wb') if sys.argv[2] != '-' else sys.stdout
    writer = MFCCWriter(out_file)

    def framecmp(f1, f2):
        x = cmp(f1.group_header.label, f2.group_header.label)
        if 0 == x: x = cmp(f1.group_header.filename, f2.group_header.filename)
        if 0 == x: x = f1.sample_offset - f2.sample_offset
        return x

    profiles, group_headers, frames = reader.read_all()
    frames.sort(cmp=framecmp)

    writer.write(profiles[0])
    for f in frames:
        writer.write(f.group_header)
        writer.write(f)

if __name__ == '__main__':
    main()


