# This script explores the features for the vqa dataset
import os
import cPickle as pickle
import numpy as np

featPath = '/srv/share/vqa/release_data/abstract_v002/scene_json/metafeatures/';

listing = os.listdir(featPath);

with open(featPath + listing[4], 'rb') as dataFile:
    print featPath + listing[4]
    feature = pickle.load(dataFile);

