#!/usr/bin/python

import struct

class MFCCReader(object):
    def __init__(self, f):
        self._f = f

        self.mel_filters, n_fft_freqs = self._read_fmt('ii')

        self.mel_freqs = self._read_fmt('%df' % self.mel_filters)
        self.fft_freqs = self._read_fmt('%df' % n_fft_freqs)

        self._framefmt = '%df' % (self.mel_filters + n_fft_freqs)
        self._framesize = struct.calcsize(self._framefmt)

    def _read_fmt(self, fmt):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self._f.read(size))

    def __iter__(self):
        return self

    def next(self):
        if self._f is None:
            raise StopIteration

        buf = self._f.read(self._framesize)
        if len(buf) < self._framesize:
            self._f.close()
            self._f = None
            raise StopIteration

        frame = struct.unpack(self._framefmt, buf)
        return frame[:self.mel_filters], frame[self.mel_filters:]
