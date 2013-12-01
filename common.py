#!/usr/bin/python

import itertools, collections, struct

class MFCCReader(object):
    ProfilePacket = collections.namedtuple('ProfilePacket', 
        ('seq',
         'frame_length', 'frame_spacing', 'sample_rate',
         'mel_filters', 'dct_length', 'fft_length',
         'mel_freqs', 'fft_freqs'))
    GroupHeaderPacket = collections.namedtuple('GroupHeaderPacket', 
        ('seq', 'profile', 'filename', 'label', 'sample_offset'))
    FramePacket = collections.namedtuple('FramePacket', 
        ('seq', 'group_header', 'mel_powers', 'fft_powers', 'dct_coeffs', 'sample_offset'))

    PROFILE_PACKET_ID = 1
    GROUP_HEADER_PACKET_ID = 2
    FRAME_PACKET_ID = 3

    def __init__(self, f):
        self._f = f
        self.current_profile = None
        self.current_group_header = None

        self._profile_seq = 0
        self._group_header_seq = 0
        self._frame_seq = 0
        self._sample_offset = None

        try:
            self._f.seek(0, 1)
            self.seekable = True
        except IOError:
            self.seekable = False

    def _read_fmt(self, fmt):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self._f.read(size))

    def __iter__(self):
        return self

    def next(self):
        packet_id = self._f.read(1)
        if len(packet_id) < 1:
            raise StopIteration
        (packet_id,) = struct.unpack('=b', packet_id)

        if MFCCReader.PROFILE_PACKET_ID == packet_id:
            frame_length, frame_spacing, sample_rate, \
            mel_filters, dct_length, fft_length = self._read_fmt('=hhhbbh')
            x = self._read_fmt('=%df' % (mel_filters + fft_length))

            self._profile_seq += 1
            self.current_profile = MFCCReader.ProfilePacket(
                seq = self._profile_seq,
                frame_length = frame_length,
                frame_spacing = frame_spacing,
                sample_rate = sample_rate,
                mel_filters = mel_filters,
                dct_length = dct_length,
                fft_length = fft_length, 
                mel_freqs = x[:mel_filters],
                fft_freqs = x[mel_filters:]
            )
            return self.current_profile

        if MFCCReader.GROUP_HEADER_PACKET_ID == packet_id:
            filename_len, label_len, sample_offset = self._read_fmt('=bbi')
            filename, label = self._read_fmt('=%ds%ds' % (filename_len, label_len))

            self._sample_offset = sample_offset
            self._group_header_seq += 1
            self.current_group_header = MFCCReader.GroupHeaderPacket(
                seq = self._group_header_seq,
                profile = self.current_profile,
                filename = filename,
                label = label,
                sample_offset = self._sample_offset
            )
            return self.current_group_header 

        if MFCCReader.FRAME_PACKET_ID == packet_id:
            profile = self.current_profile
            x = self._read_fmt('=%df' % (profile.mel_filters + profile.fft_length + profile.dct_length))
            x = iter(x)
            mel_powers = list(itertools.islice(x, profile.mel_filters))
            fft_powers = list(itertools.islice(x, profile.fft_length))
            dct_coeffs = list(itertools.islice(x, profile.dct_length))

            self._frame_seq += 1
            frame = MFCCReader.FramePacket(
                seq = self._frame_seq,
                group_header = self.current_group_header,
                mel_powers = mel_powers,
                fft_powers = fft_powers,
                dct_coeffs = dct_coeffs,
                sample_offset = self._sample_offset
            )
            self._sample_offset += self.current_profile.frame_spacing
            return frame

        raise Exception('unrecognized packet id %d' % packet_id)

    def read_all(self):
        profiles = []
        group_headers = []
        frames = []
        for packet in self:
            if isinstance(packet, MFCCReader.ProfilePacket):
                profiles.append(packet)
            if isinstance(packet, MFCCReader.GroupHeaderPacket):
                group_headers.append(packet)
            if isinstance(packet, MFCCReader.FramePacket):
                frames.append(packet)
        return (profiles, group_headers, frames)

class SeekableMFCCReader(object):
    def __init__(self, reader):
        self._reader = reader
        self.history = []
        self.hindex = 0
        self.seekable = True

    def __iter__(self):
        return self

    def get_current_frame(self):
        while self.hindex >= len(self.history):
            packet = next(self._reader)
            if isinstance(packet, MFCCReader.FramePacket):
                self.history.append(packet)
        return self.history[self.hindex]

    def get_current_group_header(self):
        return self.get_current_frame().group_header
    def get_current_profile(self):
        return self.get_current_group_header().profile

    current_group_header = property(get_current_group_header)
    current_profile = property(get_current_profile)

    def next(self):
        self.hindex += 1
        return self.get_current_frame()

    def seek(self, offs):
        self.hindex += offs
        if self.hindex < 0:
            self.hindex = 0
        try:
            return self.get_current_frame()
        except StopIteration:
            self.hindex = len(self.history)-1
            return self.history[self.hindex]

