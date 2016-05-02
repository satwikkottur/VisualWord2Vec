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

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Error in usage:')
        print('python lemmatizeVPTrain.py <path to data> <path to store>(optional)');

    dataPath = sys.argv[1];
    if len(sys.argv) > 2:
        savePath = sys.argv[2];
    else:
        # Check for existance of folders and create accordingly
        savePath = '../../data/vp/';
        if not os.path.isdir('../../data/'): os.makedirs('../../data/');
        if not os.path.isdir(savePath): os.makedirs(savePath);

    print('Extracting data and saving at %s...' % savePath);

    #def main():
    # Read the file
    readPath = '../dumps/vp_train_sentences_raw.txt';
    readFile = open(dataPath, 'rb');

    # Saving the file back
    savePath = '../dumps/vp_train_sentences_lemma.txt';
    saveFile = open(savePath, 'wb');

    tokenPath = '../dumps/vp_train_tokens_lemma.txt';
    tokenFile = open(tokenPath, 'wb');

    # tokenize
    rawTokenLines = [word_tokenize(i.strip('\n')) \
                                    for i in readFile.readlines()];
    tokenLines = [[j.lower() for j in i] \
                                     for i in rawTokenLines];
    readFile.close();

    # First tag the sentences, get wordnet tags and then lemmatize
    posLines = [pos_tag(i) for i in tokenLines];
    newTags = [[(i[0], get_wordnet_pos(i[1])) for i in j] \
                                                for j in posLines];

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

    # Save sentences
    # check for full stops
    for i in tokens:
        for j in xrange(0, len(i)-1):
            if(i[j+1] != '.'):
                saveFile.write(i[j] + ' ');
            else:
                saveFile.write(i[j]);

        # No space after full stop, line break
        print (tokens.index(i), len(i), i)
        if len(i):
            saveFile.write(i[-1] + '\n');
        else:
            saveFile.write('\n');

    saveFile.close();
