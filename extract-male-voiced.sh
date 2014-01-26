#!/bin/sh

trainingset='^(jp1m1|sw1m1|kd1m1|mr1m1|sg1m1|sg2m1|jc1m1|wm1m1|bc1m1|ms1m1|js1m1|ao1m1|rg1m1|jo1m1|dg1m1|zb1m1|kd2m1|zk1m1|ts1m1|ps1m1|sp1m1|tz1m1)'
testset='^(jk1m1|wb1m1|jp2m1|pl1m1|pw1m1)'
sounds='^[aeilnouwyz]$'

./extract-sounds.py $trainingset $sounds mfccs/corpora.mfcc mfccs/training.mfcc
./extract-sounds.py $testset $sounds mfccs/corpora.mfcc mfccs/test.mfcc
