"""
Code for approach as well as baselines to evaluate the Learning by Playing paper

"""
import numpy as np
from google_ngrams.getngrams import runQuery
import pdb
import urllib

class CommonSense():
    def __init__(self, method):
        self.method = method
        print "Using {0} method to evaluate assertions".format(method)

    def apply_model(self, test, val, source_test, source_val):

        if self.method == "val_baseline":
            score = np.zeros([len(test), 1])

            # filter out val tuples which were created randomly
            # tuples created randomly have source = 0
            val_baseline = [val_ass for i, val_ass in enumerate(val) if source_val[i] == 1]

            for i, ass in enumerate(test):

                for val_ass in val_baseline:
                    if val_ass == ass:
                        score[i] += 1

        elif self.method == "google_ngrams":
            score = np.zeros([len(test), 1])

            for i, ass in enumerate(test):
                query = ' '.join(ass)
                score[i] = runQuery(query)

        elif self.method == "our_approach":
            print "Our approach Not Implemented"
            return None

        return score
