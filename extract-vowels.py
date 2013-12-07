#!/usr/bin/python

import sys, itertools
import numpy as np
from common import *

mean_mel_shape = {
    'a' : [-34.9185523987, -29.9691905975, -31.5162124634, -32.8478736877, -33.8252983093, -30.2893657684, -27.0197219849, -28.9049167633, -33.069152832, -33.5058631897, -31.5254020691, -35.281288147, -43.0660209656, -46.6845207214, -45.8083076477, -44.3152580261, -47.1400794983, -49.0764007568, -46.719367981, -48.4097671509, -55.0743751526, -58.5036277771, -60.0622673035, -61.0283126831, -62.293510437, -64.5958023071],
    'e' : [-34.0851783752, -27.8282165527, -28.9837417603, -28.9580173492, -27.6556606293, -26.427274704, -31.7940444946, -39.9104537964, -43.7911987305, -45.8875198364, -45.4957237244, -40.9212799072, -37.4667205811, -40.1562461853, -42.1526031494, -39.4786376953, -42.4065361023, -45.6109085083, -43.6367301941, -46.0816116333, -53.0594863892, -58.4154701233, -59.5125999451, -59.5651245117, -61.1862449646, -64.3872146606],
    'i' : [-35.4911346436, -27.9373874664, -27.6388683319, -29.0107536316, -37.0513038635, -46.4026374817, -54.7504920959, -58.5725822449, -60.2225837708, -61.8978652954, -63.3399276733, -63.0371017456, -60.3949584961, -52.3102455139, -45.6652412415, -46.6164932251, -46.1152420044, -45.1194610596, -44.0915107727, -48.8131065369, -57.2163352966, -61.7566108704, -61.5912475586, -61.8673324585, -63.2836036682, -65.8952407837],
    'o' : [-34.7325782776, -28.3874378204, -29.4853610992, -29.232421875, -27.213060379, -24.7880096436, -28.3691101074, -33.8119773865, -32.7847824097, -33.23645401, -41.0830764771, -49.1791267395, -53.1311950684, -53.5260696411, -51.0956001282, -49.5073661804, -51.2948951721, -49.9951324463, -47.4072036743, -51.3620643616, -57.7217521667, -60.4713172913, -61.9295349121, -62.1775512695, -62.8071861267, -65.3556442261],
    'u' : [-34.492401123, -27.1297340393, -26.6382026672, -24.6732997894, -26.7260837555, -33.5751686096, -40.0912322998, -41.3248023987, -42.9485054016, -47.1473617554, -52.9786148071, -58.0059890747, -60.4308319092, -59.7671051025, -56.7427940369, -56.1982192993, -58.2443580627, -55.9036903381, -54.6197357178, -58.2993240356, -63.254486084, -65.3093414307, -64.4487838745, -63.9748764038, -65.0131072998, -67.2016372681],
    'y' : [-34.9702758789, -27.3469924927, -27.6042690277, -25.8252410889, -26.5698947906, -32.6714668274, -43.306427002, -48.0193557739, -50.2528915405, -51.4315757751, -50.7761497498, -46.7063827515, -42.5022468567, -43.5577087402, -44.9186630249, -42.7178344727, -46.1708984375, -47.9860610962, -45.9292373657, -49.4921798706, -56.4920883179, -61.0907974243, -60.797580719, -60.988079071, -62.4994697571, -65.6314086914],
}

mean_mel_dc = dict([ (label, sum(values[:14]) / 14.) for (label, values) in mean_mel_shape.iteritems() ])

shape_weights = np.interp(xrange(26), [0,13,17,25], [1,1,0,0])

def select_shape(frames):
    label = frames[0].group_header.label
    target_shape = mean_mel_shape[label]
    target_dc = mean_mel_dc[label]

    best = None
    besti = None
    score = 1000000000
    for i, frame in enumerate(frames):
        dc = sum(frame.mel_powers[:14]) / 14.
        diff = np.subtract(frame.mel_powers, target_shape) - dc + target_dc
        diff *= shape_weights
        s = diff.dot(diff)
        if s < score:
            score = s
            best = frame
            besti = i

    return score, best, besti

def select_stationary(frames):
    best = None
    besti = None
    score = 1000000000
    n = len(frames)

    for i,f1,f2,f3 in itertools.izip(itertools.count(1), frames, frames[1:], frames[2:]):
        d12 = np.subtract(f1.mel_powers, f2.mel_powers)
        d32 = np.subtract(f3.mel_powers, f2.mel_powers)
        s = d12.dot(d12) + d32.dot(d32) + 15. * abs(2.*i/n - 1.)

        if s < score:
            score = s
            best = f2
            besti = i

    return score, best, besti

def main():
    if len(sys.argv) != 3:
        sys.stderr.write('USAGE: extract-vowels.py [input mfcc file] [output mfcc file]\n')
        sys.exit(1)

    in_file = open(sys.argv[1], 'rb') if sys.argv[1] != '-' else sys.stdin
    reader = MFCCReader(in_file)

    out_file = open(sys.argv[2], 'wb') if sys.argv[2] != '-' else sys.stdout
    writer = MFCCWriter(out_file)

    vowels = ('a','e','o','u','i','y')

    try:
        packet = next(reader)
        while True:
            if isinstance(packet, ProfilePacket):
                writer.write(packet)
                packet = next(reader)
                continue

            if not isinstance(packet, GroupHeaderPacket) or \
               packet.label not in vowels or \
               packet.filename[3:5] != 'm1':
                packet = next(reader)
                continue

            group_header = packet

            frames = []
            while True:
                packet = next(reader)
                if not isinstance(packet, FramePacket):
                    break
                frames.append(packet)

            if len(frames) < 3:
                print 'label %s from file %s did not have three consecutive packets!' % \
                    (group_header.label, group_header.filename)
                continue

            n = len(frames)
            #score, best, besti = select_stationary(frames)
            score, best, besti = select_shape(frames)

            group_header = group_header._replace(sample_offset = best.sample_offset)
            writer.write(group_header)
            writer.write(best)

            print 'file %s, label %s: score: %f; frame %d of %d' % \
                (group_header.filename, group_header.label, score, besti, n)

    except StopIteration:
        pass

if __name__ == '__main__':
    main()

