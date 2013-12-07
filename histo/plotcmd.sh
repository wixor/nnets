#!/bin/sh
echo "set terminal pngcairo enhanced color size 1024,512"
for i in `seq 1 25`; do
    cat <<EOF
set xrange [-10:15]
set output 'dct-$i.png'
plot for [IDX=0:5] 'ch.dct' index $i+IDX*26 using 1:2 with lines lw 2 title columnheader(1)
set xrange [-50:30]
set output 'wvl-$i.png'
plot for [IDX=0:5] 'ch.wvl' index $i+IDX*26 using 1:2 with lines lw 2 title columnheader(1)
EOF
done
