"""
File to call reverb tuple wrapper and then modify the tuples as per the following rules

1. For all minor clauses, split on "is" to form relations [Mike is sad because..] -> [Mike, sad]
2. For adjective, noun sequence (happy mike) form relation [Mike, happy]
3. For all Relations, remove pronouns and articles
4. For all Primary and Secondary clauses, remove pronouns, articles and adjectives
5. Remove all instances of "is" and "are" from relations
6. Create two new relations when seeing an and: <x and y> <rel> <z> -> <x> <rel> <z> and <y> <rel> <z> [between is a
failure case] eg. between mike and jenny -> [between mike], [between jenny]
7. Drop all relations which contain a noun
8. Lemmatize all relation words

2015-03-31 01:06 rama <vrama91@vt.edu>

"""
import sys
import pdb
import re
from collections import defaultdict
from collections import Counter
import nltk
import reverb
from nltk.corpus import wordnet as wn
import nltk_to_wn as nwn
from pyInflector.inflector import Inflector

def explore_tuples(tok):
    """
    Given a set of tuples for a sentence, explore additional tuples
    :param tup:
    :param pos:
    :param dataset: string
    :return:
    """
    sent = ' '.join(tok)

    # explore tuples in minor clauses

    subOrdList = ('because', 'although', 'unless', 'however', 'since')
    create_on = (' is ', ' are ')

    occu_subord = [w for w in tok if w in subOrdList]

    new_tuples = []
    for word in occu_subord:
        clauses = sent.split(word)
        len_clauses = [len(x.split(' ')) for x in clauses]

        minor_clause = clauses[len_clauses.index(min(len_clauses))]

        word_minor_clause = minor_clause.split(' ')
        for w in word_minor_clause:
            if w in create_on:
                new_tuples.append(minor_clause.split(w))

    # explore <adjective><noun> tuples - not interesting right now.
    #if dataset != 'coco':
    #    print "oops"
    #    for i, word in enumerate(tok):
    #        if nwn.is_noun(pos[word]) and i > 1 and nwn.is_adjective(pos[tok[i-1]]):
    #            new_tuples.append([tok[i], tok[i-1]])

    return new_tuples


def filter_pos(tuples, pos, it):
    """
    For all P and S, remove <adj><pronoun><article>
    For all R, remove <pronoun><article> and word "is" and "are"
    :param tuple: list of list
    :param pos: dict
    :return:
    """
    # what to remove for P, R, S
    remove = [[0, 5, 4], [5, 4], [0, 5, 4]]


    modified_tuple = []
    for tup in tuples:
        new_tup = []
        for i, words in enumerate(tup):
            chunk = words.split(' ')
            # remove empty characters
            chunk = [w for w in chunk if len(w) > 0]
            # put an exception for cannot
            new_chunk = [w for w in chunk if nwn.penn_to_num(pos[w]) not in remove[i]]
            # remove is and are
            if 'is' in new_chunk:
                new_chunk.remove('is')
            elif 'are' in new_chunk:
                new_chunk.remove('are')

            new_tup.append(' '.join(new_chunk))

        modified_tuple.append(new_tup)

    return modified_tuple


def split_on_and(tuples):
    new_tuples = []
    for tup in tuples:
        new_chunk = []
        for i, chunk in enumerate(tup):
            if i == 0 or i == 2:
                new_chunk.append(re.split(' and | , ', chunk))
            elif i == 1:
                relation = chunk

        if len(new_chunk) == 2:
            for w1 in new_chunk[0]:
                for w2 in new_chunk[1]:
                    new_tuples.append([w1, relation, w2])

        elif len(new_chunk) == 1:
            for w1 in new_chunk[0]:
                new_tuples.append([w1, relation])

    return new_tuples


def process_tuples(tuples, pos, tokens):
    """
    Process the ReVerb Tuples to make the following changes:
    1. For all minor clauses, find <noun><is><adjective/verb> and add relation
    2. For <adjective, noun> sequence, add a relation <noun><adjective>
    3. For all P and S, remove adjectives, pronouns and articles
    4. For all R, remove pronouns, articles, and <is/are>
    5. "and" occurs in relation, create new relations
    :param tuples: dict of list of list
    :param pos: list of list
    :param dataset: string
    :param tokens: list of list
    :rtype: processed_tuples: dict of list of list
    """

    # process parts of speech form dictionary
    dic_pos = []
    for tags in pos:
        dic_pos.append({k: v for k, v in tags})

    # add relation
    for k, tok in enumerate(tokens):
        tuples[k].extend(explore_tuples(tok))

    # remove some parts of speech
    for k, tup in tuples.iteritems():
        try:
            tuples[k] = filter_pos(tup, dic_pos[k], k)
        except KeyError:
            tuples[k] = []
            # print "Skipped Tuple Due to odd formatting of input sentence"

    # create new relations when " and " occurs
    for k, tup in tuples.iteritems():
            tuples[k] = split_on_and(tup)

    # Convert plural primary or secondary noun to singular
    # also, if a noun has ',' and 'and', remove and clean
    singular_tuples = defaultdict(list)

    convert_singular = Inflector()

    dirty1 = ('group', 'couple', 'pair', 'bunch', 'crowd', 'team')
    dirty2 = ('two', 'three', 'four', 'five', 'Two', 'Three', 'Four', 'Five')
    for k, tups in tuples.iteritems():
        dic_tup_pos = dic_pos[k]
        image_tup = tuples[k]
        new_tups = []
        for tup in image_tup:
            new_tup = []
            for i, chunks in enumerate(tup):
                if i == 0 or i == 2:
                    words_split_of = chunks.split(' of ')
                    if len(words_split_of) > 1 and words_split_of[0] in dirty1:
                        if dic_tup_pos[words_split_of[0]] == 'NNPS' or dic_tup_pos[words_split_of[0]] == 'NNS':
                            new_tup.append(convert_singular.singularize(words_split_of[1]))
                        else:
                            new_tup.append(words_split_of[1])
                    else:
                        words_split = chunks.split(' ')
                        remove_list = [w for w in words_split if w in dirty2]
                        for item in remove_list:
                            words_split.remove(item)

                        mod_words_split = []
                        for w in words_split:
                            if w != '' and nwn.penn_to_wn(dic_tup_pos[w]) == wn.NOUN:
                                if dic_tup_pos[w] == 'NNPS' or dic_tup_pos[w] == 'NNS':
                                    mod_words_split.append(convert_singular.singularize(w))
                                else:
                                    mod_words_split.append(w)
                            else:
                                mod_words_split.append(w)
                        new_tup.append(' '.join(mod_words_split))
                else:
                    new_tup.append(chunks)

            new_tups.append(new_tup)
        singular_tuples[k].extend(new_tups)

    # lemmatize all the relation tuples, and remove tuples with noun in relation
    tuples = singular_tuples
    new_tuples = defaultdict(list)
    for k, tups in tuples.iteritems():
        dic_tup_pos = dic_pos[k]
        new_tups = []
        for tup in tups:
            relation = tup[1].split(' ')
            new_relation = []
            for w in relation:
                if w != '' and nwn.is_noun(dic_tup_pos[w]):
                    new_relation = []
                    break

                if w != '' and nwn.penn_to_wn(dic_tup_pos[w]) is not None:
                    new_relation.append(nltk.WordNetLemmatizer().lemmatize(w, nwn.penn_to_wn(dic_tup_pos[w])))
                else:
                    new_relation.append(nltk.WordNetLemmatizer().lemmatize(w))

            if len(new_relation) > 0:
                tup[1] = ' '.join(new_relation)
                new_tups.append(tup)
        new_tuples[k].extend(new_tups)

    # remove all tuples which have an empty primary/secondary object/relation
    # convert everything to lower case
    tuples = new_tuples
    new_tuples = defaultdict(list)

    for k, tups in tuples.iteritems():
        image_tup = tuples[k]
        new_image_tup = []
        for tup in image_tup:
            if tup[0] != '' and tup[-1] != '' and tup[1] != '':
                low_tup = [w.lower() for w in tup]
                new_image_tup.append(low_tup)
        new_tuples[k].extend(new_image_tup)

    return new_tuples


def main(json_file):
    """
    Usage: myTupleExtractor.py <json_file>

    Parameters:
    <json_file> : 	name of the json file which has sentences to run the tuple extraction on
                    file should be present in myTupleExtractor/data/

    :return: count_rels : defaultdict( str: count)
    :return: count_nouns: defaultdict( str: count)
    :return: tuples : dict of list of list

    Notes:
    Contents of json_file should have the following format:
    dict ( image_id : list of sentences)
    """
    print ""
    print json_file
    print ""

    input_file = json_file
    tup_data = reverb.TupleExtractor(input_file)

    tuples = process_tuples(tup_data.rev, tup_data.pos, tup_data.tokens)
    # get some statistics
    all_rels = []
    all_nouns = []

    for i, tups in tuples.iteritems():
        for tup in tups:
            assert isinstance(tup, list)
            for j, chunk in enumerate(tup):
                if j == 0 or j == 2:
                    all_nouns.append(chunk)
                else:
                    all_rels.append(chunk)

    count_rels = Counter(all_rels)
    count_nouns = Counter(all_nouns)

    print "Total number of relations is {0} and total number of nouns is {1}".format(len(count_rels), len(count_nouns))

    return tuples, count_rels, count_nouns


if __name__ == "__main__":
    tuples, count_rels, count_nouns = main(sys.argv[1])
