# Class to retrieve the image using text only
import os
import numpy as np
import re

class ImageRetriever:
    # Attributes
    # Word2vec embeddings
    embeds = {};
    embedDim = 0;
    vocabSize = 0;

    # Constructor
    def __init__(self):
        pass
        # Word2vec embeddings
        #self.embeds = {};
        #self.embedDim

    # Method to retrive the image
    def retrieveImage(self, imgTag):
        # Image collection path
        imgPath = '/srv/share/al/data_model/image_data/';

        imgName = imgPath + imgTag + '.png';
        # Check if existance of file
        if not os.path.isfile(imgName):
            print 'File not found!'

    # Read the word2vec embeddings
    def readWord2Vec(self, embedPath):
        # Reading the file
        with open(embedPath, 'rb') as dataFile:
            lines = [i.strip('\n') for i in dataFile.readlines()];

        # Read number of vocabsize and dimension
        regObj = re.match('(\d*) (\d*)', lines[0]);
        self.embedDim = regObj.group(1);
        self.vocabSize = regObj.group(2);

        for i in lines[1:]:
            word = i[0]
            self.embeds[word] = np.fromstring(i[1:], dtype = float, sep = ' ');

        print 'Done reading the embeddings!'


