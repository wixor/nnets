#!./python

import sys

freqs = (0.0, 68.5, 143.7, 226.2, 316.8, 416.3, 525.5, 645.4, 777.0, 921.5, 1080.1, 1254.2, 1445.4, 1655.3, 1885.7, 2138.6, 2416.3, 2721.2, 3055.9, 3423.3, 3826.7, 4269.5, 4755.7, 5289.4, 5875.3, 6518.6, 7224.7, 8000.0)
n_filters = len(freqs) - 2

def main():
    num = [0.0] * n_filters
    denom = [0.0] * n_filters

    first = True
    for line in sys.stdin:
        if first:
            first = False
            continue

        freq, power = line.strip().split('\t')
        freq = float(freq)
        power = float(power)

        for i in xrange(n_filters):
            low = freqs[i]
            mid = freqs[i+1]
            high = freqs[i+2]
            if freq > low and freq < high:
                if freq < mid:
                    factor = (freq - low) / (mid - low)
                else:
                    factor = (high - freq) / (high - mid)
                num[i] += factor * power
                denom[i] += factor

    for freq, x,y in zip(freqs[1:], num, denom):
        print '%f %f' % (freq, x / y)

if __name__ == '__main__':
    main()

