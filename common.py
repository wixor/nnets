#!./python

import sys, itertools, collections, struct
import numpy as np
try:
    import scipy.fftpack
except ImportError:
    pass # maybe we won't need it...
import wavelet

ProfilePacket = collections.namedtuple('ProfilePacket', 
    ('seq',
     'frame_length', 'frame_spacing', 'sample_rate', 'mel_power_threshold',
     'mel_filters', 'fft_length', 'mel_freqs', 'fft_freqs'))
GroupHeaderPacket = collections.namedtuple('GroupHeaderPacket', 
    ('seq', 'profile', 'filename', 'label', 'sample_offset'))
FramePacket = collections.namedtuple('FramePacket', 
    ('seq', 'group_header', 'mel_powers', 'fft_powers', 'sample_offset'))

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
            frame_length, frame_spacing, sample_rate, \
            mel_power_threshold = self._read_fmt('=bHHHHf')

            x = self._read_fmt('=%df' % (mel_filters+2 + fft_length))

            self._profile_seq += 1
            self.current_profile = ProfilePacket(
                seq = self._profile_seq,
                frame_length = frame_length,
                frame_spacing = frame_spacing,
                sample_rate = sample_rate,
                mel_power_threshold = mel_power_threshold,
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
            x = self._read_fmt('=%df' % (profile.mel_filters + profile.fft_length))
            x = iter(x)
            mel_powers = list(itertools.islice(x, profile.mel_filters))
            fft_powers = list(itertools.islice(x, profile.fft_length))

            self._frame_seq += 1
            frame = FramePacket(
                seq = self._frame_seq,
                group_header = self.current_group_header,
                mel_powers = mel_powers,
                fft_powers = fft_powers,
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
            '=bbHHHHf%df' % (packet.mel_filters+2 + packet.fft_length),
            PROFILE_PACKET_ID,
            packet.mel_filters, packet.fft_length,
            packet.frame_length, packet.frame_spacing, packet.sample_rate,
            packet.mel_power_threshold,
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
        x = packet.mel_powers + packet.fft_powers
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

## ------------------------------------------------------------------------- ##

wavelet_matrix_cache = dict()
def get_wavelet_matrix(n):
    M = []
    for i in xrange(n):
        A = [0.] * n
        A[i] = 1.
        wavelet.forward(A)
        M.append(A)
    M = np.matrix(M, dtype=np.float32)
    wavelet_matrix_cache[n] = M
    return M

pca_matrix = np.matrix([
 [ -8.48677478e-02, 4.48216035e-01, 1.32214492e-01, -3.52124473e-01, 1.59988117e-01, -3.75733739e-01, 1.47300114e-01, -9.78533198e-02, 4.67698530e-01, -2.05567044e-01, 2.29368909e-01, -2.10150833e-01, 1.31653698e-01, -2.90933620e-02, 2.79265046e-02, -2.89819695e-02, -5.46065007e-03, -1.62133646e-01, -2.94567039e-02, 2.61887597e-03, -2.60637031e-07],
 [ -8.93112848e-02, 4.07818252e-01, 7.60240103e-02, -1.44458952e-01, 8.68222234e-03, -9.48834857e-02, 2.89902306e-02, 3.56490083e-02, -8.72939342e-02, 9.82927524e-02, -2.53536848e-01, 2.88575581e-01, -2.97602716e-01, 1.44045443e-01, -1.41664685e-01, 2.19243735e-01, -5.13417619e-02, 6.22652759e-01, 8.70402111e-02, 7.14463972e-03, 3.77736730e-06],
 [ -1.13174116e-01, 3.81745763e-01, 2.57829618e-02, -6.13584301e-02, -7.74654281e-02, 4.07303548e-02, -6.33515527e-02, 3.58127184e-02, -2.87628714e-01, 2.16339448e-01, -2.78272985e-01, 3.52988695e-01, -4.96394794e-02, -4.91958316e-02, 2.12574238e-01, -1.30309728e-01, 9.01379504e-02, -6.02400599e-01, -8.23327978e-02, -6.68089994e-03, -3.48907323e-06],
 [ -1.14975815e-01, 3.47550421e-01, -2.37118260e-02, 1.51959157e-01, -1.79831718e-01, 2.55548553e-01, -4.38872735e-02, 4.63966908e-02, -2.94331741e-01, 1.66062562e-01, 1.59262687e-01, -3.46605655e-01, 5.82463940e-01, -7.25732417e-02, 1.09981090e-02, -1.36220503e-01, -7.54511958e-02, 2.49103782e-01, 4.72270470e-02, 1.00510676e-02, -1.73445572e-07],
 [ -2.12914779e-02, 2.26939968e-01, -3.49514897e-02, 4.13663950e-01, -1.86680675e-01, 3.33391504e-01, 8.52657091e-02, -1.60379159e-02, 8.75371359e-02, -1.91124892e-01, 2.54504329e-01, -1.66948963e-01, -3.76381826e-01, 2.19821921e-01, -3.06595376e-01, 2.81880978e-01, 1.24496865e-01, -2.35813259e-01, -5.47508908e-02, -1.32630201e-02, 4.59953282e-07],
 [ 1.35027826e-01, 5.10377341e-02, -5.97367814e-02, 4.56966681e-01, -7.93133166e-02, 9.85033821e-02, 1.78924493e-01, -8.12422750e-02, 3.64797641e-01, -1.89711875e-01, -1.31567438e-01, 1.93055212e-01, -9.33446209e-02, -2.01025711e-01, 4.22286082e-01, -3.78336816e-01, -1.84990264e-01, 1.81186208e-01, 5.66582847e-02, 7.19031329e-03, -7.96243312e-07],
 [ 2.84529367e-01, -8.18206436e-02, -1.03474775e-01, 2.91982964e-01, 6.08083104e-02, -2.37445053e-01, 1.62251247e-01, -1.02265001e-01, 1.90330896e-01, 2.27778212e-01, -3.91607568e-01, 1.48953554e-02, 3.79277337e-01, -1.26808771e-01, -2.16013119e-01, 3.98982434e-01, 2.30761703e-01, -6.43057162e-02, -8.30104725e-02, -1.27447589e-02, 3.85016134e-06],
 [ 3.32695414e-01, -4.79809074e-02, -1.38305928e-01, 1.15554425e-01, 1.48107753e-01, -3.55300399e-01, 8.84785124e-02, -2.63276879e-02, -1.89085004e-01, 3.38775057e-01, 8.33637418e-02, -2.22174765e-01, -2.03966852e-01, 3.74673616e-01, -1.36824905e-01, -3.52638411e-01, -2.85115480e-01, -9.36942183e-02, 1.66299983e-01, 2.44882339e-02, -5.05678424e-06],
 [ 3.37016290e-01, 1.06967612e-02, -1.88072066e-01, -2.92142561e-03, 1.51478366e-01, -1.82526372e-01, -9.81767977e-02, 8.53665111e-02, -3.21571402e-01, -9.11100837e-02, 4.04878391e-01, 2.36861624e-02, -1.47729883e-01, -2.30233914e-01, 3.49388770e-01, 1.53085000e-01, 3.29702560e-01, 1.59271486e-01, -3.07135151e-01, -7.19720102e-02, 4.11032943e-06],
 [ 3.48246056e-01, 1.15105985e-02, -1.65989301e-01, -1.55748560e-01, 1.18551036e-01, 1.40475939e-01, -2.74151827e-01, 1.38685081e-01, -7.88750217e-02, -4.10279397e-01, 2.38734807e-02, 2.30857565e-01, 1.59314258e-01, -1.62235366e-01, -2.28539629e-01, 1.18223568e-01, -2.93263037e-01, -1.33211291e-01, 4.16865577e-01, 1.64720882e-01, -4.52562114e-06],
 [ 3.27461999e-01, -6.38086316e-02, 9.64784609e-03, -3.20671863e-01, -7.66160526e-03, 3.60561704e-01, -2.17033303e-01, 2.47818473e-02, 1.79581888e-01, -9.36252378e-02, -2.75119230e-01, -1.12448538e-01, 5.24965332e-02, 3.09486242e-01, -7.49315394e-02, -2.51898948e-01, 1.03309235e-01, 7.58477900e-02, -4.19447303e-01, -2.66562794e-01, 4.80050085e-06],
 [ 2.26920753e-01, -1.69075712e-01, 2.87734411e-01, -3.01090482e-01, -2.11665700e-01, 2.69575903e-01, 9.41397983e-02, -1.07088771e-01, 1.15917734e-01, 3.24427825e-01, 2.57991154e-02, -2.06022179e-01, -1.59211445e-01, -7.73498763e-02, 2.91573898e-01, 1.79051509e-01, 1.34931999e-01, -1.24418606e-02, 3.07802603e-01, 3.71133256e-01, -5.92104543e-06],
 [ 6.95806504e-02, -2.20356060e-01, 4.65136442e-01, -7.14092762e-02, -2.72560781e-01, -4.93380566e-02, 2.64436966e-01, -2.35624672e-04, -1.13857115e-01, 4.64548710e-02, 2.30492397e-01, 2.21627957e-01, 2.75988523e-02, -2.40606099e-01, -2.11065451e-01, 1.69011842e-02, -2.88791256e-01, -3.24077934e-02, -1.06716736e-01, -4.62803776e-01, 7.09821962e-06],
 [ -9.69196267e-02, -2.05745217e-01, 4.30282204e-01, 1.51781928e-01, -9.06172332e-02, -2.39208959e-01, 2.28790335e-02, 2.79613864e-01, -1.54934611e-01, -3.42119336e-01, -5.92308562e-02, 7.57739181e-02, 1.50446927e-01, 2.74879627e-01, -3.77256638e-02, -1.61281816e-01, 2.29229306e-01, 3.32880404e-02, -1.09025663e-01, 4.54826217e-01, -7.14468417e-06],
 [ -2.21865360e-01, -1.19282368e-01, 2.23267535e-01, 1.75549096e-01, 1.63268410e-01, -1.26552525e-01, -4.31252052e-01, 2.95235491e-01, 5.12228182e-02, -1.94013732e-02, -2.14207265e-01, -3.47782546e-01, -1.24878586e-01, -4.18907611e-02, 2.39475444e-01, 1.89813760e-01, -7.25950447e-02, -4.75165429e-02, 2.34161254e-01, -3.86683158e-01, 5.49997272e-06],
 [ -2.14404587e-01, -1.08302174e-01, 1.19149336e-01, 1.19436416e-01, 3.14452288e-01, 9.25672296e-02, -4.40044390e-01, -1.78626571e-01, 2.02872173e-01, 3.26533256e-01, 2.10161520e-01, 1.55214747e-01, -6.81607489e-02, -2.36626511e-01, -2.47742898e-01, -1.10825318e-01, -1.10930090e-01, 3.95972119e-02, -2.82660299e-01, 2.98525656e-01, -4.30403714e-06],
 [ -2.00992987e-01, -1.78792906e-01, 3.94063144e-02, 1.40212565e-02, 3.67744929e-01, 1.55051847e-01, 9.33393771e-02, -5.28784558e-01, -1.29999323e-01, -1.04137318e-01, 1.24623938e-01, 2.15588700e-01, 1.85180330e-01, 3.44263490e-01, 1.43886734e-01, 3.17954693e-02, 2.03694737e-01, 1.50037259e-02, 2.75049650e-01, -2.21398335e-01, 3.23786161e-06],
 [ -2.37415988e-01, -1.57001212e-01, -1.40906812e-01, -1.33698835e-01, 2.99607544e-01, 1.45632547e-01, 4.09472393e-01, -5.54252916e-02, -2.69215711e-01, -1.99500259e-01, -2.87996679e-01, -3.10799420e-01, -1.39532756e-01, -2.21898506e-01, 1.04346222e-03, 8.67494286e-02, -2.90657095e-01, -3.37350257e-02, -2.62720453e-01, 1.70966796e-01, -1.24394620e-06],
 [ -2.36208992e-01, -1.62392767e-01, -2.94245959e-01, -1.29786505e-01, 1.10340302e-01, 1.17468132e-01, 2.68067185e-01, 5.06196426e-01, 1.68358670e-01, 1.67337721e-01, 1.05743119e-01, 1.00534225e-01, -3.25449995e-02, -1.33929201e-01, -2.03121586e-01, -2.55787218e-01, 3.73981343e-01, 2.91107741e-02, 2.34121200e-01, -1.17821275e-01, -1.00130266e-06],
 [ -2.29604085e-01, -2.13933259e-01, -3.75699684e-01, -9.96870536e-02, -2.88277441e-01, -7.51633688e-02, -3.52997568e-02, 1.56545462e-01, 1.77538024e-01, 1.03099852e-01, 1.61059699e-01, 1.82878721e-01, 1.58009400e-01, 3.56829368e-01, 2.75045096e-01, 3.09612798e-01, -3.48448636e-01, -2.85754034e-02, -1.88286098e-01, 7.26556635e-02, 1.80522158e-06],
 [ -2.00446289e-01, -1.57023675e-01, -2.83550930e-01, -1.17960017e-01, -5.08955381e-01, -2.73355137e-01, -2.40348106e-01, -4.10396083e-01, -7.90629330e-02, -1.68524741e-01, -1.21592457e-01, -1.32743940e-01, -1.33447363e-01, -2.00532554e-01, -1.69973485e-01, -1.79059137e-01, 1.86798811e-01, 4.11735798e-02, 1.00316759e-01, -2.43915742e-02, -7.22767598e-07]
], dtype=np.float32)

def xmaker(mode):
    if 'mels' == mode:
        return lambda f : np.array(f.mel_powers, dtype=np.float32)

    if 'pca' == mode:
        def fn(f):
            mels = np.array(f.mel_powers, dtype=np.float32)
            return np.array(mels * pca_matrix[:, 0:10]).ravel()
        return fn

    if 'dcts' == mode:
        def fn(f):
            mels = np.array(f.mel_powers, dtype=np.float32)
            return scipy.fftpack.dct(mels, type=2)[1:11]
        return fn

    if 'wvls' == mode:
        def fn(f):
            mels = np.array(f.mel_powers, dtype=np.float32)
            n = len(f.mel_powers)
            M = wavelet_matrix_cache[n] if n in wavelet_matrix_cache else get_wavelet_matrix(n)
            return np.array(mels * M[:, 1:11]).ravel()
        return fn

    raise ValueError('unrecognized mode; must be mels, pca, dcts or wvls')

def is_clipped(f):
    threshold = 0.5 + f.group_header.profile.mel_power_threshold
    return any([ v < threshold for v in f.mel_powers ])

