#!/bin/sh
letter=$1

../mel-filters.py < spectrum-${letter}.txt > mels-${letter}.txt

gnuplot <<EOF
set terminal pngcairo enhanced size 1024,256
set output 'plot-${letter}.png'
plot 'spectrum-${letter}.txt' using 1:2 with lines lc 7, 'mels-${letter}.txt' using 1:2 with linespoints ls 7 lw 3
EOF

