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
 [ -1.09950180e-01, 4.19035849e-01, 1.60880846e-01, -4.46030248e-01, 1.99268158e-01, -2.46569278e-01, 1.77429596e-01, -1.58002957e-01, 4.48015121e-01, -1.71340116e-01, 1.96274203e-01, -2.39370919e-01, 1.45062879e-01, -2.76522323e-02, 4.36673712e-02, 2.94920288e-02, -4.23526430e-02, 1.75054657e-01, -9.23141001e-04, 2.14866012e-03, 2.27539977e-07],
 [ -1.04132063e-01, 3.91812014e-01, 9.41518555e-02, -1.89392803e-01, 4.73457613e-02, -6.73985272e-02, 3.63126204e-02, 4.54004115e-02, -9.10507965e-02, 9.78773285e-02, -2.32436202e-01, 3.13950000e-01, -2.90711449e-01, 1.39496873e-01, -1.42274474e-01, -2.12781542e-01, 4.89597044e-02, -6.24437088e-01, -2.85859999e-02, 5.89803929e-03, 8.62026567e-07],
 [ -1.07903130e-01, 3.71897346e-01, 3.30209008e-02, -4.01507886e-02, -3.59481252e-02, 4.94631083e-02, -6.24627748e-02, 7.69201616e-02, -2.96609810e-01, 2.42395327e-01, -2.50032992e-01, 3.75596588e-01, -2.49616874e-02, -4.30647618e-02, 2.09561133e-01, 1.32601306e-01, -5.55421349e-03, 6.09126102e-01, 2.69840741e-02, -8.84243878e-03, -1.07601241e-06],
 [ -8.71873917e-02, 3.60239431e-01, -1.95041216e-02, 1.93143889e-01, -1.81266739e-01, 2.09366723e-01, -8.74359914e-02, 1.00926098e-01, -2.86422143e-01, 1.60961038e-01, 1.59106134e-01, -3.54509234e-01, 5.67759607e-01, -8.38433311e-02, -1.61461560e-02, 1.34176655e-01, -1.03329896e-02, -2.75538144e-01, 1.03268217e-02, 9.03409789e-03, -3.02086255e-07],
 [ 1.92793777e-02, 2.72555556e-01, -1.99608197e-02, 3.90484849e-01, -2.58257472e-01, 2.36458440e-01, -7.83190729e-03, -1.34858398e-02, 9.29784901e-02, -2.31039114e-01, 2.46968250e-01, -1.64389382e-01, -3.93560174e-01, 2.28621383e-01, -2.96256576e-01, -2.92549858e-01, 6.38054792e-02, 2.58862092e-01, -1.61520405e-02, -9.97794653e-03, 3.19076700e-07],
 [ 1.73736051e-01, 1.01507269e-01, -4.23638529e-02, 4.15145718e-01, -1.60757914e-01, 2.82480580e-02, 1.22596435e-01, -1.32928006e-01, 3.66327855e-01, -2.13648029e-01, -1.51737067e-01, 1.74379410e-01, -1.05069436e-01, -2.01395630e-01, 4.17833780e-01, 3.87844170e-01, -1.42908224e-01, -2.17583804e-01, 2.59362441e-02, 9.56758854e-03, -2.31734286e-07],
 [ 3.10928889e-01, -5.67257118e-02, -8.33633018e-02, 2.47428578e-01, 2.18970677e-02, -2.60898091e-01, 2.00166720e-01, -1.20838434e-01, 1.91056795e-01, 2.16307287e-01, -3.97061494e-01, 3.93708935e-03, 3.41672666e-01, -1.47113972e-01, -2.32676397e-01, -4.01573632e-01, 2.23137955e-01, 1.08155860e-01, -6.87871749e-02, -1.15277377e-02, 7.85928959e-07],
 [ 3.43856583e-01, -4.71824001e-02, -1.16365985e-01, 5.94220389e-02, 1.37493682e-01, -3.47469818e-01, 1.60814464e-01, 2.92528980e-03, -1.67223736e-01, 3.41238820e-01, 9.45904222e-02, -2.12861932e-01, -2.05704377e-01, 3.76032064e-01, -1.47113170e-01, 3.50534330e-01, -2.99711478e-01, 9.36025469e-03, 1.77862723e-01, 2.98289881e-02, -1.22201102e-06],
 [ 3.38695407e-01, -1.57849204e-03, -1.75694394e-01, -5.22599091e-02, 1.56964390e-01, -2.00591638e-01, -3.94057210e-02, 1.12732159e-01, -2.84319720e-01, -8.71751653e-02, 4.24160947e-01, 2.31074866e-02, -1.41780681e-01, -2.14287598e-01, 3.58279912e-01, -1.36898353e-01, 3.55206569e-01, -4.57410187e-02, -3.26851955e-01, -8.84402158e-02, 1.67226424e-06],
 [ 3.43251055e-01, -1.59173874e-02, -1.68046588e-01, -1.68782767e-01, 1.43725084e-01, 1.09584169e-01, -2.69661084e-01, 1.17614083e-01, -8.82983036e-02, -4.13264323e-01, 3.38388423e-02, 2.35766511e-01, 1.73329324e-01, -1.66172356e-01, -2.03156887e-01, -1.28047792e-01, -3.01347433e-01, 1.16173160e-02, 4.25931737e-01, 1.88042573e-01, -2.64480486e-06],
 [ 3.10732715e-01, -1.03265625e-01, 4.05319710e-03, -2.86074151e-01, 1.74246957e-02, 3.92978350e-01, -2.54387062e-01, -1.60372752e-02, 1.17882583e-01, -9.18232894e-02, -2.92243538e-01, -9.81776938e-02, 6.44042782e-02, 3.07098631e-01, -8.82610809e-02, 2.43893322e-01, 9.43823204e-02, 1.68844894e-02, -4.10138999e-01, -2.88254049e-01, 3.24632543e-06],
 [ 1.90294105e-01, -2.06553332e-01, 2.96660398e-01, -2.56451708e-01, -2.07688647e-01, 3.51542724e-01, 5.99581842e-02, -9.06796352e-02, 7.56433968e-02, 3.39587129e-01, 2.36537044e-02, -1.86643847e-01, -1.60997718e-01, -7.30506077e-02, 2.79494240e-01, -1.71180607e-01, 1.46445458e-01, -2.47692707e-02, 2.89688328e-01, 3.74303282e-01, -3.83597703e-06],
 [ 2.08775355e-02, -2.41099728e-01, 4.77204160e-01, -5.85574998e-02, -2.69731081e-01, -1.63272445e-02, 2.53202415e-01, 6.97675026e-02, -8.13310793e-02, 3.20227231e-02, 2.43832026e-01, 2.28731443e-01, 4.91096897e-02, -2.37280166e-01, -2.02102151e-01, -1.72719199e-02, -3.04778523e-01, 6.04010930e-03, -1.01488729e-01, -4.45159351e-01, 4.73537266e-06],
 [ -1.24380440e-01, -2.05659307e-01, 4.16107957e-01, 1.18318328e-01, -6.18959506e-02, -2.78190425e-01, 5.88839571e-03, 2.89189544e-01, -1.01057584e-01, -3.47716854e-01, -7.89623434e-02, 5.49977037e-02, 1.52757476e-01, 2.84569285e-01, -2.64792656e-02, 1.68455973e-01, 2.52206852e-01, 2.85679367e-02, -1.04059601e-01, 4.39952274e-01, -5.04157368e-06],
 [ -2.03777722e-01, -1.20200719e-01, 1.89306689e-01, 1.62216421e-01, 2.04082598e-01, -1.89133215e-01, -4.40233060e-01, 2.33857057e-01, 5.76613441e-02, 2.08446995e-03, -2.24436395e-01, -3.41594587e-01, -1.41796541e-01, -5.01954728e-02, 2.47794432e-01, -2.00732739e-01, -9.19919160e-02, -1.00684333e-02, 2.43897167e-01, -3.81551877e-01, 4.06204993e-06],
 [ -1.92903709e-01, -1.16402994e-01, 7.84792851e-02, 1.76421965e-01, 3.14464361e-01, 3.96033636e-02, -4.32418737e-01, -2.41170868e-01, 1.89234272e-01, 3.07724843e-01, 2.31398482e-01, 1.60609050e-01, -6.38200954e-02, -2.22554732e-01, -2.51817499e-01, 1.17554975e-01, -8.99645892e-02, -7.81234236e-03, -2.90801813e-01, 2.92113382e-01, -2.27567887e-06],
 [ -1.99712106e-01, -1.76669701e-01, 5.67499690e-03, 1.26991736e-01, 3.07410316e-01, 1.65086647e-01, 1.50932243e-01, -5.23562623e-01, -1.67013503e-01, -1.06707865e-01, 1.40230356e-01, 1.99168120e-01, 1.87882009e-01, 3.37514994e-01, 1.30853223e-01, -4.00624418e-02, 2.09716218e-01, -3.42425177e-02, 2.85992361e-01, -2.30153082e-01, 1.52553215e-06],
 [ -2.38077952e-01, -1.42098662e-01, -1.45132032e-01, -1.17435267e-02, 2.76714997e-01, 2.15807412e-01, 4.14302107e-01, -1.81382501e-02, -2.82053588e-01, -1.94684165e-01, -2.69579649e-01, -2.91574924e-01, -1.48203964e-01, -2.16423891e-01, 1.69555576e-02, -8.89063088e-02, -3.08321239e-01, 3.06364451e-02, -2.76895788e-01, 1.85404060e-01, -4.36147274e-07],
 [ -2.33470475e-01, -1.42561241e-01, -3.10682494e-01, -5.12627444e-02, 1.16017840e-01, 1.72023904e-01, 2.34038790e-01, 4.85535509e-01, 2.54639097e-01, 1.17847679e-01, 7.99221589e-02, 7.32099536e-02, -6.62622629e-02, -1.48840465e-01, -2.22030136e-01, 2.58824515e-01, 3.73034266e-01, -8.72671568e-04, 2.38642384e-01, -1.24146945e-01, -1.17310199e-06],
 [ -2.27492216e-01, -1.91276670e-01, -3.92240202e-01, -9.36056871e-02, -2.66923466e-01, -6.32711388e-02, -2.92954447e-02, 1.67152115e-01, 2.20672363e-01, 1.26552659e-01, 1.40461956e-01, 1.73040555e-01, 1.92985099e-01, 3.59479893e-01, 2.84710815e-01, -2.96922151e-01, -3.28315916e-01, -7.03335719e-03, -1.82599724e-01, 7.82155334e-02, 1.20241207e-06],
 [ -2.22664333e-01, -1.49855497e-01, -2.82186495e-01, -2.35261692e-01, -5.00339556e-01, -3.00313526e-01, -1.92510188e-01, -3.87176040e-01, -1.68731053e-01, -1.27200384e-01, -1.17947800e-01, -1.27371391e-01, -1.32094642e-01, -2.00937908e-01, -1.60836672e-01, 1.63550068e-01, 1.58684342e-01, -6.20661438e-03, 8.20231250e-02, -2.64548359e-02, -3.99401020e-07]
], dtype=np.float32)

def xmaker(mode):
    if 'mels' == mode:
        return lambda f : np.array(f.mel_powers, dtype=np.float32)

    if 'pca' == mode:
        def fn(f):
            mels = np.array(f.mel_powers, dtype=np.float32)
            return np.array(mels * pca_matrix[:, 0:7]).ravel()
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

