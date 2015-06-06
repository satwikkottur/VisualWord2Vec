from nltk.corpus import wordnet as wn

def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']

def is_pronoun(tag):
    return tag in ['PRP', 'PRP$']

def is_article(tag):
    return tag in ['DT']

def is_conj(tag):
    return tag in ['CC']

def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return None

def penn_to_num(tag):
    if is_adjective(tag):
        return 0
    elif is_noun(tag):
        return 1
    elif is_adverb(tag):
        return 2
    elif is_verb(tag):
        return 3
    elif is_article(tag):
        return 4
    elif is_pronoun(tag):
        return 5
    else:
        return -1
