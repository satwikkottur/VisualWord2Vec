# Class to find the odd one out
import sys
sys.path.append('../utils/');
from Word2Vec import loadWord2Vec, computeEmbedding
import numpy as np

class OddFinder:
    visEmbed = {};
    w2vEmbed = {};
    embedDim = 0;

    def readEmbeddings(self, visPath, w2vPath):
        (self.visEmbed, self.embedDim, _) = loadWord2Vec(visPath);
        (self.w2vEmbed, _, _) = loadWord2Vec(w2vPath);

    def findOutOne(self, tuples, embeds):
        oddId = [];
        # Compute the embeddings each of the members
        for i in tuples:
            feats = [computeEmbedding(j, embeds, self.embedDim) for \
                                j in i];

            # Compute the mean and distance from the mean
            sumFeat = np.sum(feats);
            otherMean = [(sumFeat - j)/(len(feats)-1) for j in feats];
            #dist = [np.linalg.norm(otherMean - j) for j in feats];
            meanFeat = np.mean(feats);
            dist = [np.linalg.norm(meanFeat - j) for j in feats];

            # Odd one out if very far away
            oddId.append(dist.index(max(dist)));

        return oddId;
        
    def performTask(self, data):
        # Perform the smae task twice
        w2vOdd = self.findOutOne(data, self.w2vEmbed);
        visOdd = self.findOutOne(data, self.visEmbed);

        # Print results
        self.printOddOne(data, visOdd, w2vOdd);

    def printOddOne(self, data, oddIdVis, oddIdW2v):
        assert len(oddIdVis) == len(oddIdW2v) == len(data)

        for i in xrange(len(data)):
            if oddIdVis[i] != oddIdW2v[i]:
                print "%-60s  (%-20s) {%-20s}" % \
                    (str(data[i]), data[i][oddIdVis[i]], data[i][oddIdW2v[i]])
