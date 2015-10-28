# Script to lemmatize tokens using nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

# Read the file
readPath = '../dumps/token_dump.txt';
readFile = open(readPath, 'rb');

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
saveFile.close();
