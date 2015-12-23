#!/usr/bin/env python
from ast import literal_eval
from pandas import DataFrame  # http://github.com/pydata/pandas
import re
import requests               # http://github.com/kennethreitz/requests
import subprocess
import sys

count = []
count += [13588391]
count += [314843401]
count += [977069902]
count += [1313818354]
count += [1176470663]

corpora = dict(eng_us_2012=17, eng_us_2009=5, eng_gb_2012=18, eng_gb_2009=6,
               chi_sim_2012=23, chi_sim_2009=11, eng_2012=15, eng_2009=0,
               eng_fiction_2012=16, eng_fiction_2009=4, eng_1m_2009=1,
               fre_2012=19, fre_2009=7, ger_2012=20, ger_2009=8, heb_2012=24,
               heb_2009=9, spa_2012=21, spa_2009=10, rus_2012=25, rus_2009=12,
               ita_2012=22)



def getNgrams(query, corpus, startYear, endYear, smoothing, caseInsensitive):
    params = dict(content=query, year_start=startYear, year_end=endYear,
                  corpus=corpora[corpus], smoothing=smoothing,
                  case_insensitive=caseInsensitive)
    if params['case_insensitive'] is False:
        params.pop('case_insensitive')
    if '?' in params['content']:
        params['content'] = params['content'].replace('?', '*')
    if '@' in params['content']:
        params['content'] = params['content'].replace('@', '=>')
    req = requests.get('http://books.google.com/ngrams/graph', params=params)
    res = re.findall('var data = (.*?);\\n', req.text)

    if res:
        data = sum({qry['ngram']: count[len(qry['ngram'].split())-1]*sum(qry['timeseries'])/float(len(qry['timeseries']))
                for qry in literal_eval(res[0])}.values())
    else:
        data = 0

    return req.url, params['content'], data


def runQuery(query):

    corpus, startYear, endYear, smoothing = 'eng_2012', 1800, 2000, 3
    printHelp, caseInsensitive, allData = False, False, False

    url, urlquery, data = getNgrams(query, corpus, startYear, endYear,
                                      smoothing, caseInsensitive)
    return data

if __name__ == '__main__':
    argumentString = ' '.join(sys.argv[1:])
    try:
        print runQuery(argumentString)
    except RuntimeError:
        print('An error occurred.')
