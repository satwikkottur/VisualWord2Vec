# Script to read the tuples for the text based image retrieval task
# and clean them up
import pickle
import os
from nltk.stem import WordNetLemmatizer

# Setting up paths
picklePath = '/home/satwik/VisualWord2Vec/data/text_retrieval_pilot.p';
tuplesData = pickle.load(open(picklePath, 'rb'));

# Order : Relation, Primary, Secondary
tuples = tuplesData['data'];

lmt = WordNetLemmatizer();
for i in tuples.keys():
    # Get tuples and lemmatize
    # For each of R, P, S:
    #   1. Split by ' ' to get each component
    #   2. Convert into lower case
    #   3. Lemmatize using 'v' for R, 'n' for P,S
    #   4. Join back using ' '
    lemmaTuple = [(' '.join([lmt.lemmatize(c.lower(), 'v') for c in j[0].split(' ')]), \
                    ' '.join([lmt.lemmatize(c.lower(), 'n') for c in j[1].split(' ')]), \
                    ' '.join([lmt.lemmatize(c.lower(), 'n') for c in j[2].split(' ')])) \
                    for j in tuples[i]];

    # Save into the main structure
    tuplesData['data'][i] = lemmaTuple;

# Save the pickle file
dumpPath = '/home/satwik/VisualWord2Vec/data/text_retrieval_pilot_lemma.p';
pickle.dump(tuplesData, open(dumpPath, 'wb'));
