"""
Created on : 4/11/15 11:20 PM by rama

Main script for running assertion classification on a test set, reporting ROC curves etc

"""

# load the val and test datasets
import initData
import curateData
from commonSense import commonSense
from sklearn.metrics import precision_recall_curve, auc
import pylab as pl
import numpy as np
import pdb
import pickle

dataset = initData.LoadData('cs_val.json', 'cs_test.json')

# vals[0] holds val assertions/tuples, vals[1] holds relations, vals[2] holds nouns
val, old_val, score_val = dataset.loadDataset('val')
# test[0] holds assertions, test[1] holds labels
test, old_test, score_test = dataset.loadDataset('test')


assert(isinstance(val, list))
assert(isinstance(test, list))
assert(isinstance(score_val, list))
assert(isinstance(old_val, list))
# get labels from scores
test_labels = []
for scr in score_test:
    test_labels.append(int(scr > 0))
# see how many times test labels and old_test differ

# options: val_baseline, text_baseline, our_approach
# NOTE: text_baseline and our_method need to be implemented
methods = ["val_baseline"]

scorers = [commonSense.CommonSense(method) for method in methods]

scores = []
for scorer, method in zip(scorers, methods):
    print "Using method {0} to evaluate".format(method)
    # add some small noise to the val baseline scores, to get a continuous ROC curve
    if method == "val_baseline":
        output = scorer.apply_model(test, val, old_test, old_val)
        for i, out in enumerate(output):
            output[i] += np.random.uniform(0, 0.001)
        scores.append(output)
    else:
        scores.append(scorer.apply_model(test, val))
    pickle.dump(scores[-1], open(method + '.p', 'w'))

# plot ROC curves by varying a threshold
for score, method in zip(scores, methods):
    if score is not None:
        precision, recall, thresholds = precision_recall_curve(test_labels, score)
        area = auc(recall, precision)
        print "Area Under Curve: %0.2f" % area

        pl.clf()
        pl.plot(recall, precision, label='Precision-Recall curve')
        pl.xlabel('Recall')
        pl.ylabel('Precision')
        pl.ylim([0.0, 1.05])
        pl.xlim([0.0, 1.0])
        pl.title('Precision-Recall for method {0}: AUC=%0.2f'.format(method) % area)
        pl.legend(loc="lower left")
        pl.show()
