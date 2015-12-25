# Script to read the cliparts, sentences and then align based on mutual 
# information
import cPickle as pickle

# Input:
#   1. List of training scenes (20000, currently)
#   2. Type of scene for each of the training scenes
#   3. List of cliparts present in the each of the training scenes
#   4. Tuples for each of the caption sentences
#   5. Image map between the captions and scenes
# 
# Output:
#   1. Alignment between the clipart and the words

class Aligner:
    def __init__():
        # Read the scene-caption map, 
        # scene types, scene clipart, 
        # tuples for the image
        # Add data path
        dataPath = '/home/satwik/VisualWord2Vec/data/vqa/';

        with open(dataPath + 'vqa_feature_map.txt', 'r') as fileId:
            self.maps = [i.strip('\n') for i in fileId.readlines()];
        with open(dataPath + 'scene_type.cPickle', 'r') as fileId:
            self.types = pickle.load(fileId);
        with open(dataPath + 'clipart_occurance.cPickle', 'r') as fileId:
            self.cliparts = pickle.load(fileId);
        with open(dataPath + 'vqa_train_tuples.pickle','r') as fileId:
            self.tuples = pickle.load(fileId);

    

if __name__ == '__main__':
    Aligner();
    wrapper();
