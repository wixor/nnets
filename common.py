#!./python

import sys, itertools, collections, struct
import numpy as np

ProfilePacket = collections.namedtuple('ProfilePacket', 
    ('seq',
     'frame_length', 'frame_spacing', 'sample_rate',
     'mel_filters', 'fft_length', 'mel_freqs', 'fft_freqs'))
GroupHeaderPacket = collections.namedtuple('GroupHeaderPacket', 
    ('seq', 'profile', 'filename', 'label', 'sample_offset'))
FramePacket = collections.namedtuple('FramePacket', 
    ('seq', 'group_header', 'mel_powers', 'fft_powers', 'dct_coeffs', 'wvl_coeffs', 'sample_offset'))

PROFILE_PACKET_ID = 1
GROUP_HEADER_PACKET_ID = 2
FRAME_PACKET_ID = 3

class MFCCReader(object):
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

        if PROFILE_PACKET_ID == packet_id:
            mel_filters, fft_length, \
            frame_length, frame_spacing, sample_rate = self._read_fmt('=bHHHH')

            x = self._read_fmt('=%df' % (mel_filters+2 + fft_length))

            self._profile_seq += 1
            self.current_profile = ProfilePacket(
                seq = self._profile_seq,
                frame_length = frame_length,
                frame_spacing = frame_spacing,
                sample_rate = sample_rate,
                mel_filters = mel_filters,
                fft_length = fft_length, 
                mel_freqs = x[:mel_filters+2],
                fft_freqs = x[mel_filters+2:]
            )
            return self.current_profile

        if GROUP_HEADER_PACKET_ID == packet_id:
            filename_len, label_len, sample_offset = self._read_fmt('=bbi')
            filename, label = self._read_fmt('=%ds%ds' % (filename_len, label_len))

            self._sample_offset = sample_offset
            self._group_header_seq += 1
            self.current_group_header = GroupHeaderPacket(
                seq = self._group_header_seq,
                profile = self.current_profile,
                filename = filename,
                label = label,
                sample_offset = self._sample_offset
            )
            return self.current_group_header 

        if FRAME_PACKET_ID == packet_id:
            profile = self.current_profile
            x = self._read_fmt('=%df' % (3*profile.mel_filters + profile.fft_length))
            x = iter(x)
            mel_powers = list(itertools.islice(x, profile.mel_filters))
            fft_powers = list(itertools.islice(x, profile.fft_length))
            dct_coeffs = list(itertools.islice(x, profile.mel_filters))
            wvl_coeffs = list(itertools.islice(x, profile.mel_filters))

            self._frame_seq += 1
            frame = FramePacket(
                seq = self._frame_seq,
                group_header = self.current_group_header,
                mel_powers = mel_powers,
                fft_powers = fft_powers,
                dct_coeffs = dct_coeffs,
                wvl_coeffs = wvl_coeffs,
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
            if isinstance(packet, ProfilePacket):
                profiles.append(packet)
            if isinstance(packet, GroupHeaderPacket):
                group_headers.append(packet)
            if isinstance(packet, FramePacket):
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
            if isinstance(packet, FramePacket):
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

class MFCCWriter(object):
    def __init__(self, f):
        self._f = f

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

    def _write_profile(self, packet):
        self._f.write(struct.pack(
            '=bbHHHH%df' % (packet.mel_filters+2 + packet.fft_length),
            PROFILE_PACKET_ID,
            packet.mel_filters, packet.fft_length,
            packet.frame_length, packet.frame_spacing, packet.sample_rate,
            *(packet.mel_freqs + packet.fft_freqs)
        ))

    def _write_group_header(self, packet):
        self._f.write(struct.pack(
            '=bbbi%ds%ds' % (len(packet.filename), len(packet.label)),
            GROUP_HEADER_PACKET_ID,
            len(packet.filename), len(packet.label), packet.sample_offset,
            packet.filename, packet.label
        ))

    def _write_frame(self, packet):
        x = packet.mel_powers + packet.fft_powers + packet.dct_coeffs + packet.wvl_coeffs
        self._f.write(struct.pack('=b%df' % len(x), FRAME_PACKET_ID, *x))

    def write(self, packet):
        if isinstance(packet, ProfilePacket):
            return self._write_profile(packet)
        if isinstance(packet, GroupHeaderPacket):
            return self._write_group_header(packet)
        if isinstance(packet, FramePacket):
            return self._write_frame(packet)
        raise TypeError('unsupported packet type ' + type(packet))

def oread(filename):
    return open(filename, 'rb') if filename != '-' else sys.stdin
def owrite(filename):
    return open(filename, 'wb') if filename != '-' else sys.stdout

def xmaker(mode):
    if 'mels' == mode:
        def fn(f):
            ret = np.array(f.mel_powers, dtype=np.float32)
            return ret - np.average(ret)
        return fn
    if 'dcts' == mode:
        return lambda f : np.array(f.dct_coeffs[1:11], dtype=np.float32)
    if 'wvls' == mode:
        return lambda f : np.array(f.wvl_coeffs[1:11], dtype=np.float32)
    raise ValueError('unrecognized mode; must be mels, dcts or wvls')

