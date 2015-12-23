"""
Class to load simple sentences to preprocess, and run Reverb on it.

2015-03-31 01:04 rama <vrama91@vt.edu>

Rewriting code after previous files got accidentally deleted
"""

import pdb
import json
import os
import pickle
import nltk
import re
from collections import defaultdict
import subprocess

def run_reverb(tokens):
    """
    Run the reverb tuple extraction code on the input sentences
    """
    reverb_input = 'reverb_in.txt'
    reverb_executable = "./reverb.sh"
    path_to_reverb = "reverb-1.0 2/"
    reverb_options = " -q "
    f = open(path_to_reverb + reverb_input, 'w')

    for sentences in tokens:
        f.write(sentences + "\n")
    f.close()

    reverb_out = subprocess.Popen(reverb_executable + reverb_options + reverb_input, cwd=path_to_reverb,
                                  shell=True, stdout=subprocess.PIPE).stdout.readlines()

    reverb_tuples_raw = defaultdict(list)
    reverb_sentence_raw = dict()

    for line in reverb_out:
        spl_line = line.strip().split('\t')
        if spl_line[0] == "extraction":
            reverb_tuples_raw[int(spl_line[2])].append(spl_line[3:6])
        elif spl_line[0] == "sentence":
            reverb_sentence_raw[int(spl_line[2])] = re.sub('[^A-Za-z0-9]', '', spl_line[3])

    curridx = 0
    keylist = sorted(reverb_tuples_raw.keys())
    reverb_tuples = defaultdict(list)

    for s in range(0, len(tokens)):
        # noinspection PyBroadException
        try:
            if reverb_sentence_raw[keylist[curridx]] == re.sub('[^A-Za-z0-9]', '', tokens[s]):
                reverb_tuples[s] = reverb_tuples_raw[keylist[curridx]]
                curridx += 1
            elif reverb_sentence_raw[keylist[curridx+1]] == re.sub('[^A-Za-z0-9]', '', tokens[s]):
                reverb_tuples[s] = reverb_tuples_raw[keylist[curridx+1]]
                curridx += 2
            elif reverb_sentence_raw[keylist[curridx+2]] == re.sub('[^A-Za-z0-9]', '', tokens[s]):
                reverb_tuples[s] = reverb_tuples_raw[keylist[curridx+2]]
                curridx += 3

            else:
                print "rip"

        except:
            print "rip"

    return reverb_tuples


class TupleExtractor():

    def __init__(self, input_file):

        out_file = input_file.split('.')[0] + 'reverb' + '.p'
        tup_file = input_file.split('.')[0] + 'tuples' + '.p'

        path_to_ss = input_file
        path_to_pos = tup_file
        path_to_rev = out_file

        data = json.loads(open(path_to_ss).read())
        sentences = [sent for key in data for sent in data[key]]

        print "Dataset has {0} sentences".format(len(sentences))

        tokens = []
        simple_sent = []
        pos = []

        for line in sentences:
            # Ensure every sentence ends with a full stop, and pick only 1 sentence in case two provided
            line = line.split('.')[0].rstrip() + '.'
            # Ensure every sentence starts with a capital letter
            line = line[0].upper() + line[1:]
            # Don't need any pre-processing, will add an exception clause in myTupleExtractor
            tokens.append(nltk.word_tokenize(line))
            simple_sent.append(line)

        print "Running Part-Of-Speech (POS) tagging"
        if os.path.exists(path_to_pos):
            pos = pickle.load(open(path_to_pos, 'r'))
        else:
            # get POS tags on tokens
            for tok in tokens:
                pos.append(nltk.pos_tag(tok))
            pickle.dump(pos, open(path_to_pos, 'w'))

        print "Running Reverb Tuple Extraction"
        if os.path.exists(path_to_rev):
            rev = pickle.load(open(path_to_rev, 'r'))
        else:
            # run reverb
            rev = run_reverb(simple_sent)
            pickle.dump(rev, open(path_to_rev, 'w'))


        self.tokens = tokens
        self.simple_sent = simple_sent
        self.pos = pos
        self.rev = rev