#!./python
import sys
import cPickle as pickle
import numpy as np
import scipy.optimize
from common import *

### ----------------------------------------------------------------------- ###

def prepend_ones(M):
    return np.vstack(( np.ones(M.shape[1]), M ))

def softmax(A):
    def f(col):
        col = col - col.max()
        col = np.exp(col)
        return np.maximum(col / col.sum(), 1e-50)
    return np.matrix(np.apply_along_axis(f, 0, A))

def softmax_crossentropy(C, Y):
    return (
        (np.multiply(np.log(C), -Y)).sum(), # value
        C - Y # derivative
    )

def nnet(Warray, X,Y, justAnswer=False):
    inputs = X.shape[0]
    outputs = Y.shape[0]
    
    W = np.matrix(Warray.reshape((outputs,inputs+1)))

    L0 = X
    A1 = W * prepend_ones(L0)
    C = softmax(A1)
    if justAnswer:
        return C
    LOSS, dLOSS_dA1 = softmax_crossentropy(C, Y)
    #dLOSS_dL0 = (W.T * dLOSS_dA1)[1:,] # strip derivative wrt. constant bias

    dLOSS_dW = dLOSS_dA1 * prepend_ones(L0).T

    print LOSS
    return LOSS, np.array(dLOSS_dW).ravel()

### ----------------------------------------------------------------------- ###

def run_classifier(X,Y,W):
    dummyY = np.zeros(Y.shape)

    C = nnet(W, X, dummyY, justAnswer=True)
    return map(int, np.apply_along_axis(np.argmax, 0, C))

def check_classifier(X,Y,W):
    ans = run_classifier(X,Y,W)

    histo = dict([ ((i,j),0) for i in xrange(Y.shape[0]) for j in xrange(Y.shape[0]) ])
    errcnt = 0

    for i in xrange(Y.shape[1]):
        a = ans[i]
        corr = np.argmax(Y[:,i])
        histo[(corr,a)] += 1
        if a != corr:
            errcnt += 1

    return Y.shape[1], errcnt, histo

### ----------------------------------------------------------------------- ###

def recognize():
    if len(sys.argv) != 4:
        sys.stderr.write('USAGE: nnet.py recognize [mfcc file] [weights file]\n')
        sys.exit(1)

    mfcc_file = oread(sys.argv[2])
    reader = MFCCReader(mfcc_file)

    with open(sys.argv[3], 'rb') as f:
        W = pickle.load(f)
        labelnames = W['labelnames']
        mode = W['mode']
        W = W['weights']

    makeX = xmaker(mode)

    Y = np.matrix(np.zeros(len(labelnames))).T

    for packet in reader:
        if isinstance(packet, ProfilePacket):
            print '\t'.join(labelnames)
            continue
        if isinstance(packet, GroupHeaderPacket):
            print '# label %s (file %s, offset %d)' % (packet.label, packet.filename, packet.sample_offset)
            continue

        X = np.matrix(makeX(packet)).T

        C = nnet(W, X, Y, justAnswer=True)
        C = [ str(float(C[i])) for i in xrange(C.shape[0]) ]
        print '\t'.join(C)

def test():
    if len(sys.argv) != 4:
        sys.stderr.write('USAGE: nnet.py test [test file] [weights file]\n')
        sys.exit(1)

    with open(sys.argv[2], 'rb') as f:
        test = pickle.load(f)
        mode = test['mode']
        X = test['X']
        Y = test['Y']
        labelnames = test['labelnames']

    with open(sys.argv[3], 'rb') as f:
        W = pickle.load(f)
        if W['mode'] != mode:
            raise ValueError('mode mismatch; test file has %s, but nnet file has %s' % (mode, W['mode'])
        if W['labelnames'] != labelnames:
            raise ValueError('label names mismatch')
        W = W['weights']

    total, errcnt, histo = check_classifier(X,Y,W)
    print 'made %d errors out of %d; accuracy %.1f%%' % (errcnt, total, 100. - 100.*errcnt/total)

    for label in labelnames:
        sys.stdout.write('\t' + label)
    print
    for i,label in enumerate(labelnames):
        sys.stdout.write(label)
        total = errcnt = 0
        for j in xrange(len(labelnames)):
            sys.stdout.write('\t%d' % histo[(i,j)] )
            total += histo[(i,j)]
            if i != j:
                errcnt += histo[(i,j)]
        print '\t(%.1f%%)' % (100. - 100.*errcnt/total)


def learn():
    if len(sys.argv) != 4:
        sys.stderr.write('USAGE: nnet.py learn [training file] [output weights file]\n')
        sys.exit(1)

    with open(sys.argv[2], 'rb') as f:
        training = pickle.load(f)
    
    mode = training['mode']
    X = training['X']
    Y = training['Y']
    inputs = X.shape[0]
    outputs = Y.shape[0]

    W = np.random.rand(outputs*(inputs+1))
    W = W * 0.6 - 0.3

    W, value, info = scipy.optimize.fmin_l_bfgs_b(nnet, W, args=(X,Y), factr=1e11)
    print 'loss: %f' % value
    print 'weights:\n%r' % W
    print 'notes:\n%r' % info
    print

    with open(sys.argv[4], 'wb') as f:
        pickle.dump(dict(
            weights = W,
            labelnames = training['labelnames'],
            mode = mode),
            f, -1)
    print 'dumped weights to %s' % sys.argv[4]

def main():
    if len(sys.argv) >= 2:
        if sys.argv[1] == 'learn':
            return learn()
        if sys.argv[1] == 'test':
            return test()
        if sys.argv[1] == 'recognize':
            return recognize()

    sys.stderr.write('USAGE: nnet.py [learn|test|recognize] [options...]\n')
    sys.exit(1)

if __name__ == '__main__':
    main()

