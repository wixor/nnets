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

    profiles, group_headers, frames = reader.read_all()
    frames.sort(key = lambda f: (f.group_header.label, f.group_header.filename, f.sample_offset))

    writer.write(profiles[0])
    last_group_header = None
    for f in frames:
        if f.group_header != last_group_header:
            writer.write(f.group_header)
            last_group_header = f.group_header
        writer.write(f)

if __name__ == '__main__':
    main()


