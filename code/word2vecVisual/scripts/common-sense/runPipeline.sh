#!/bin/bash

# Training google word2vec
nohup /usr/local/MATLAB/R2014a/bin/matlab -nodesktop -nodisplay -nosplash -r "run('/home/satwik/VisualWord2Vec/code/word2vecVisual/scripts/commonSenseTextDemo.m')" < /dev/null > output_word2vec_after 2> errors_word2vec_after
