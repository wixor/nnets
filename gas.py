#!./python

import sys, itertools, collections, multiprocessing, time
import numpy as np
import cPickle as pickle
from common import *

def pythag(M):
    return np.sqrt(np.sum(M*M, axis=1))

def classify(X,F):
    if 1 == X.ndim:
        Dmag = pythag(X - F)
        idx = np.argmin(Dmag)
        return idx, Dmag[idx]
    else:
        idx = np.empty(X.shape[0], dtype=np.int32)
        best = np.empty(X.shape[0], dtype=np.float32)
        best.fill(np.inf)
        for i in xrange(F.shape[0]):
            Dmag = pythag(X - F[i])
            idx[Dmag < best] = i
            best = np.minimum(best, Dmag)
        return idx, best

### ----------------------------------------------------------------------- ###

def analyze():
    if len(sys.argv) != 4:
        sys.stderr.write('USAGE: gas analyze [input gas file] [training mfcc file]\n')
        sys.exit(1)

    with open(sys.argv[2], 'rb') as f:
        data = pickle.load(f)
        mode = data['mode']
        F = data['F']
        del data

    with oread(sys.argv[3]) as in_file:
        frames = MFCCReader(in_file).read_all()[2]

    makeX = xmaker(mode)
    X = np.vstack(map(makeX, frames))

    Ds = np.empty((F.shape[0], F.shape[0]))
    for i in xrange(F.shape[0]):
        Ds[i] = pythag(F[i] - F)

    print '# distances'
    for i in xrange(F.shape[0]):
        for j in xrange(F.shape[0]):
            sys.stdout.write('%f ' % Ds[i,j])
        print
    
    idx, dist = classify(X,F)
    eh = [ [] for i in xrange(F.shape[0]) ]
    for i,d in itertools.izip(idx, dist):
        eh[i].append(d)

    m = dist.min()
    M = dist.max()
    bins = 30
    for i in xrange(F.shape[0]):
        eh[i],edges = np.histogram(eh[i], bins, (m,M))

    midpoints = .5 * (edges[:-1] + edges[1:])
    weights = midpoints ** (F.shape[1]-1)
    for i in xrange(F.shape[0]):
        eh[i] = eh[i].astype(np.float32)
        #eh[i] /= weights
        eh[i] /= eh[i].sum()
        
    print '\n\n# error histograms'
    for j in xrange(bins):
        sys.stdout.write('%f' % midpoints[j])
        for i in xrange(F.shape[0]):
            sys.stdout.write(' %f' % eh[i][j])
        print

### ----------------------------------------------------------------------- ###

def score():
    if len(sys.argv) != 5:
        sys.stderr.write('USAGE: gas score [input gas file] [label] [training mfcc file]\n')
        sys.exit(1)

    with open(sys.argv[2], 'rb') as f:
        data = pickle.load(f)
        makeX = xmaker(data['mode'])
        F = data['F']
        E = data['E']
        del data

    label = sys.argv[3]

    in_file = oread(sys.argv[4])
    reader = MFCCReader(in_file)

    framehits = np.zeros(F.shape[0], dtype=np.int32)
    grouphits = np.zeros(F.shape[0], dtype=np.int32)
    colours = np.zeros(F.shape[0], dtype=np.int32)
    proxim2 = np.zeros(F.shape[0], dtype=np.float32)
    proxim = np.zeros(F.shape[0], dtype=np.float32)

    c = 0
    for packet in reader:
        if isinstance(packet, GroupHeaderPacket):
            c += 1
            proxim2 += proxim
            proxim.fill(np.inf)
        if not isinstance(packet, FramePacket):
            continue
        if packet.group_header.label != label:
            continue

        X = makeX(packet)
        proxim = np.fmin(proxim, pythag(X-F))
        idx, dist = classify(X, F)

        print 'file %s, offset %d;  class %d dist %f' % (
            packet.group_header.filename, packet.sample_offset,
            idx, dist)

        framehits[idx] += 1
        if colours[idx] != c:
            colours[idx] = c
            grouphits[idx] += 1

    proxim2 += proxim
    proxim2 /= float(c)

    perm = np.argsort(proxim2)

    print 'selected group: %d' % perm[0]
    print '  class         err          frames          groups          proxim'
    for i in perm:
        print '%7d %15.5f %15d %15d %15.5f' % (
            i, E[i], framehits[i], grouphits[i], proxim2[i] )

### ----------------------------------------------------------------------- ###

class BestGroupFilter(object):
    __slots__ = ('frame','dist')

    def __init__(self):
        self.frame = None
        self.dist = np.inf

    def consider(self, frame, dist):
        if dist < self.dist:
            self.frame = frame
            self.dist = dist

    def flush(self, writer):
        if self.frame is None:
            return

        frame = self.frame
        group_header = frame.group_header
        print 'file %s, offset %d: distance %f' % (group_header.filename, frame.sample_offset, self.dist)

        writer.write(group_header._replace(sample_offset = frame.sample_offset))
        writer.write(frame)

        self.frame = None
        self.dist = np.inf

def filter():
    if len(sys.argv) != 6:
        sys.stderr.write('USAGE: gas filter [input gas file] [class1,class2,...] [input mfcc file] [output mfcc file]\n')
        sys.exit(1)

    with open(sys.argv[2], 'rb') as f:
        data = pickle.load(f)
        makeX = xmaker(data['mode'])
        F = data['F']
        del data

    classes = [ int(x.strip()) for x in sys.argv[3].split(',') ]

    in_file = oread(sys.argv[4])
    reader = MFCCReader(in_file)

    out_file = owrite(sys.argv[5])
    writer = MFCCWriter(out_file)

    best = BestGroupFilter()

    for packet in reader:
        if isinstance(packet, ProfilePacket):
            writer.write(packet)
        elif isinstance(packet, GroupHeaderPacket):
            best.flush(writer)
        else:
            #dist = np.min(pythag(makeX(packet) - F[classes]))
            idx, dist = classify(makeX(packet), F)
            if idx in classes:
                best.consider(packet, dist)

    best.flush(writer)

    out_file.close()
    in_file.close()

### ----------------------------------------------------------------------- ###

def gas(X, ncount, steps, eps_range, lamb_range, debug=None):
    F = X[np.random.choice(X.shape[0], ncount, replace=False)]

    for step in xrange(0, steps):
        f = 1. * step / (steps-1)
        eps = eps_range[0] * (eps_range[1] / eps_range[0]) ** f
        lamb = lamb_range[0] * (lamb_range[1] / lamb_range[0]) ** f

        sel = np.random.randint(0, X.shape[0])
        D = X[sel] - F
        Dmag = np.sum(D*D, axis=1)
        for k,i in enumerate(np.argsort(Dmag)):
            u = eps * np.exp(-1. * k / lamb)
            F[i] += u * D[i]
            if u < 1e-3:
                break

        if debug is not None and 0 == (step+1) % debug:
            idx, dist = classify(X,F)
            loss = np.average(dist)
            print '%d %f %f %f' % (step+1, eps, lamb, loss)

    return F

def learn_worker(arg):
    X, ncount = arg

    print '# [%d] training... (got %d frames)' % (ncount, X.shape[0])
    F = gas(X, ncount=ncount, steps = 100000, eps_range = (0.5, 0.05), lamb_range = (50., .5))

    print '# [%d] analyzing....' % ncount
    idx, dist = classify(X, F)

    E = np.zeros(F.shape[0], dtype=np.float32)
    count = np.zeros(F.shape[0], dtype=np.int32)
    for i,d in itertools.izip(idx, dist):
        E[i] += d
        count[i] += 1
    E /= count

    perm = np.argsort(E)
    F = F[perm]
    E = E[perm]

    print '# [%d] average error: %f; median error: %f\n' \
          '# [%d] cluster size deviation: %f' % (
                ncount, np.average(dist), np.median(dist),
                ncount, np.std(count) )
    sys.stdout.flush()
    
    return F, E

def learn():
    if len(sys.argv) != 5:
        sys.stderr.write('USAGE: gas.py learn [mode] [training mfcc file] [output gas file]\n')
        sys.exit(1)

    print '# reading input...'
    mode = sys.argv[2]
    makeX = xmaker(mode)

    with oread(sys.argv[3]) as in_file:
        X = []
        for packet in MFCCReader(in_file):
            if isinstance(packet, FramePacket):
                X.append(makeX(packet))
        X = np.vstack(X)

    #F, E = learn_worker( (X,200) )

    pool = multiprocessing.Pool(processes=4)
    pool.map(learn_worker, [(X,ncount) for ncount in xrange(10,151,2)], 1)
    pool.close()
    pool.join()
    return

    with owrite(sys.argv[4]) as f:
        pickle.dump(dict(mode = mode, F = F, E = E), f, -1)

def main():
    if len(sys.argv) >= 2:
        if sys.argv[1] == 'learn':
            return learn()
        if sys.argv[1] == 'analyze':
            return analyze()
        if sys.argv[1] == 'score':
            return score()
        if sys.argv[1] == 'filter':
            return filter()

    sys.stderr.write('USAGE: gas.py [learn|analyze|score|filter]\n')
    sys.exit(1)

if __name__ == '__main__':
    main()

