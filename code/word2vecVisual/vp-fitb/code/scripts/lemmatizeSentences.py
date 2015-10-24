# Script to lemmatize tokens using nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wordnet
from nltk.stem import WordNetLemmatizer

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

def main(readPath, savePath, tokenPath):
    # Read the file
    readFile = open(readPath, 'rb');
    # Saving the file back
    saveFile = open(savePath, 'wb');
    tokenFile = open(tokenPath, 'wb');

    # tokenize
    rawTokenLines = [word_tokenize(i.strip('\n')) \
                                    for i in readFile.readlines()];
    tokenLines = [[j.lower() for j in i] \
                                     for i in rawTokenLines];
    readFile.close();

    # First tag the sentences, get wordnet tags and then lemmatize
    print('Beginning to tag...')

    # Check for unicode and ignore in case of errors
    #posLines = [];
    #for i in xrange(0, len(tokenLines)):
        #print (i, len(tokenLines))
        #posLines.append(pos_tag([unicode(j, errors = 'ignore') for j in tokenLines[i]]));
    posLines = [pos_tag([unicode(j, errors = 'ignore') for j in i])\
                                for i in tokenLines];

    #posLines = [pos_tag(i) for i in tokenLines];
    newTags = [[(i[0], get_wordnet_pos(i[1])) for i in j] \
                                                for j in posLines];
    print('Done tagging...')

    # Store all the tokens after lemmatizing
    tokenSet = [];
    # Lemmatize
    lmt = WordNetLemmatizer();
    tokens = [[lmt.lemmatize(i[0], i[1]) for i in j] \
                                            for j in newTags];
    [tokenSet.extend(i) for i in tokens]
    tokenSet = list(set(tokenSet));
    [tokenFile.write(i + '\n') for i in tokenSet];
    tokenFile.close();

    # Save tokens

    # Save sentences
    # check for full stops
    for i in tokens:
        for j in xrange(0, len(i)-1):
            if(i[j+1] != '.'):
                saveFile.write(i[j] + ' ');
            else:
                saveFile.write(i[j]);

        # No space after full stop, line break
        # print (tokens.index(i), len(i), i)
        if len(i):
            saveFile.write(i[-1] + '\n');
        else:
            saveFile.write('\n');

    saveFile.close();

###############################################################################
# VP training dataset
'''readPath = '../dumps/vp_train_sentences_raw.txt';
savePath = '../dumps/vp_train_sentences_lemma.txt';
tokenPath = '../dumps/vp_train_tokens_lemma.txt';
main(readPath, savePath, tokenPath);'''

# MS COCO training set
'''readPath = '/home/satwik/VisualWord2Vec/data/coco_train_minus_cs_test.txt';
savePath = '/home/satwik/VisualWord2Vec/data/coco_train_minus_cs_test_lemma.txt';
tokenPath = '/home/satwik/VisualWord2Vec/data/coco_train_minus_cs_test_lemma_tokens.txt';
main(readPath, savePath, tokenPath);'''

# VP all sentences
'''readPath = '../dumps/vp_sentences_1.txt';
savePath = '../dumps/vp_sentences1_lemma.txt';
tokenPath = '../dumps/vp_sentence1_tokens_lemma.txt';
main(readPath, savePath, tokenPath);

readPath = '../dumps/vp_sentences_2.txt';
savePath = '../dumps/vp_sentences2_lemma.txt';
tokenPath = '../dumps/vp_sentence2_tokens_lemma.txt';
main(readPath, savePath, tokenPath);'''

# VQA sentences
readPath = '/home/satwik/VisualWord2Vec/data/vqa_train_captions_raw.txt';
savePath = '/home/satwik/VisualWord2Vec/data/vqa_train_captions_lemma.txt';
tokenPath = '/home/satwik/VisualWord2Vec/data/vqa_train_captions_tokens.txt';
main(readPath, savePath, tokenPath);

###############################################################################
'''
# Read the tokens, lemmatize
lmt = WordNetLemmatizer();
rawTokens = [i.strip('\n') for i in readFile.readlines()];
tokens = [lmt.lemmatize(i) for i in rawTokens];

#for i in xrange(0, len(tokens)):
#    print (rawTokens[i], tokens[i]);

# Writing back the lemmatized tokens
savePath = '../dumps/lemmatized_token_dumps.txt';
saveFile = open(savePath, 'wb');

for i in tokens:
    saveFile.write(i + '\n')

readFile.close();
saveFile.close();'''
'''posLines = [];
for i in tokenLines:
print i
unicodeLine = [unicode(j, errors = 'ignore') for j in i];
posLines.append(pos_tag(unicodeLine));'''
