# Script to lemmatize tokens using nltk
from nltk import word_tokenize
import nltk
from nltk.tag.perceptron import PerceptronTagger
from nltk.corpus import wordnet as wordnet
from nltk.stem import WordNetLemmatizer
import sys
import pdb

# Code adapted from stackoverflow
def progress(count, total, suffix=''):
    bar_len = 20
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # Return noun by default, as lemmatized has default noun
        return wordnet.NOUN

#-------------------------------------------------------------------------
# Given the file to read in filePath, and file to save in savePath
# (optionally a tokenPath to save the tokens)
def lemmatizeSentences(filePath, savePath, tokenPath=None):
    print('Tokenizing %s and saving at %s...' % (filePath, savePath));

    tagger = PerceptronTagger()

    # Read the file
    inputFileId = open(filePath, 'r');

    # tokenize, removing full stops
    print('\nTokenizing..')
    tokenLines = [];

    lines = [i.strip('\n').lower() \
                            for i in inputFileId.readlines()];
    for lineId, line in enumerate(lines):
        if lineId % 10 == 0: progress(lineId, len(lines));
        tokenLines.append(word_tokenize(line));
    inputFileId.close();

    # First tag the sentences, get wordnet tags and then lemmatize
    print('\nTagging POS..')
    posLines = []; newTags = [];
    for lineId, line in enumerate(tokenLines):
        if lineId % 10 == 0: progress(lineId, len(tokenLines));

        taggedLine = nltk.tag._pos_tag(line, None, tagger)
        posLines.append(taggedLine);
        newTags.append([(i[0], get_wordnet_pos(i[1])) for i in taggedLine]);

    # Store all the tokens after lemmatizing
    tokenSet = [];
    tokens = [];
    # Lemmatize
    print('\nLemmatizing..')
    lmt = WordNetLemmatizer();
    for lineId, tags in enumerate(newTags):
        if lineId % 10 == 0: progress(lineId, len(newTags));

        tokens.append([lmt.lemmatize(i[0], i[1]) for i in tags]);

    # Collect all the tokens for saving if tokenPath is not None
    if tokenPath is not None:
        tokenFile = open(tokenPath, 'wb');
        [tokenSet.extend(i) for i in tokens]
        tokenSet = list(set(tokenSet));
        [tokenFile.write(i + '\n') for i in tokenSet];
        tokenFile.close();

    # Saving the tokenized, lemmatized sentences
    saveFile = open(savePath, 'w');

    # check for full stops
    for iterId, sentenceTokens in enumerate(tokens):
        # Print progress occasionally
        if iterId % 10 == 0: progress(iterId, len(tokens));

        # Replace space + fullstop with just full stop
        saveFile.write(' '.join(sentenceTokens).replace(' .', '.') + '\n');

    saveFile.close();

#-------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Error in usage:')
        print('python lemmatizeVPTrain.py <path to data> <path to store>');

    dataPath = sys.argv[1];
    savePath = sys.argv[2];

    print('Extracting data and saving at %s...' % savePath);
    lemmatizeSentences(dataPath + 'vp_train_sentences_raw.txt',\
                        savePath + 'vp_train_full.txt');
    lemmatizeSentences(dataPath + 'vp_sentences_1.txt',\
                        savePath + 'vp_sentences1_lemma.txt');
    lemmatizeSentences(dataPath + 'vp_sentences_2.txt',\
                        savePath + 'vp_sentences2_lemma.txt');
    print('\n Done processing!')
