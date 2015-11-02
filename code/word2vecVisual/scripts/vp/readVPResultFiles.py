# This is a script to read the output files from the vp task
import re
    
def printResults(dumpPath):
    #dumpPath = '../modelsNdata/backup/out_val_sents';
    rootPath = '/home/satwik/VisualWord2Vec/code/word2vecVisual/';
    #rootPath = '/home/satwik/VisualWord2Vec/code/word2vecVisual/modelsNdata/backup/';
    fileId = open(rootPath + dumpPath, 'rb');
    if(fileId == None):
        print 'File not found!'
        return

    lines = [i.strip('\n') for i in fileId.readlines()];

    regExp = 'mAP: Test \(([\d.]*)\) Val\(([\d.]*)\)';
    splitLines = [re.search(regExp, i) for i in lines];

    valScores = [i.group(2) for i in splitLines if i is not None];
    testScores = [i.group(1) for i in splitLines if i is not None];

    print dumpPath
    print 'Baseline: %s\nBest VP: %s\n' % \
                (testScores[0], testScores[valScores.index(max(valScores[0:100]))])
    fileId.close();

###########################################################################
#dumpPath = '../modelsNdata/backup/out_val_sents';
'''dumpPaths = ['out_val1k_sents', \
            'out_val1k_sents_pca', \
            'out_val1k_words', \
            'out_val1k_words_pca', \
            'out_val1k_words_wiki', \
            'out_val1k_sents_wiki'];

dumpPaths.extend(['out_enum_test_sents', \
            'out_enum_test_desc', \
            'out_enum_test_words', \
            'out_enum_test_winds']);'''

'''dumpPaths.extend(['out_enum_wiki_desc', \
                    'out_enum_wiki_desc_pca', \
                    'out_enum_wiki_sents', \
                    'out_enum_wiki_sents_pca', \
                    'out_enum_wiki_words', \
                    'out_enum_wiki_words_pca', \
                    'out_enum_wiki_winds', \
                    'out_enum_wiki_winds_pca']);'''

dumpPaths = [];
'''dumpPaths.extend(['out_vp_50_descs_pca', \
                'out_vp_50_sents', \
                'out_vp_50_sents_pca', \
                'out_vp_50_descs', \
                'out_vp_50_winds', \
                'out_vp_50_winds_pca', \
                'out_vp_50_words', \
                'out_vp_50_words_pca']);

dumpPaths.extend(['out_vp_100_descs_pca', \
                'out_vp_100_sents', \
                'out_vp_100_sents_pca', \
                'out_vp_100_descs', \
                'out_vp_100_winds', \
                'out_vp_100_winds_pca', \
                'out_vp_100_words', \
                'out_vp_100_words_pca']);'''

dumpPaths.extend(['out_vp_wiki_50_descs_pca', \
                'out_vp_wiki_50_sents', \
                'out_vp_wiki_50_sents_pca', \
                'out_vp_wiki_50_descs', \
                'out_vp_wiki_50_winds', \
                'out_vp_wiki_50_winds_pca', \
                'out_vp_wiki_50_words', \
                'out_vp_wiki_50_words_pca']);

dumpPaths.extend(['out_vp_wiki_100_descs_pca', \
                'out_vp_wiki_100_sents', \
                'out_vp_wiki_100_sents_pca', \
                'out_vp_wiki_100_descs', \
                'out_vp_wiki_100_winds', \
                'out_vp_wiki_100_winds_pca', \
                'out_vp_wiki_100_words', \
                'out_vp_wiki_100_words_pca']);

'''dumpPaths.extend(['out_vp_coco_descs', \
                    'out_vp_coco_descs_pca', \
                    'out_vp_coco_sents', \
                    'out_vp_coco_sents_pca', \
                    'out_vp_coco_words', \
                    'out_vp_coco_words_pca', \
                    'out_vp_coco_winds', \
                    'out_vp_coco_winds_pca']);'''

'''dumpPaths.extend(['out_wiki_desc', \
                    'out_wiki_descs_pca', \
                    'out_wiki_sents', \
                    'out_wiki_sents_pca', \
                    'out_wiki_words', \
                    'out_wiki_words_pca', \
                    'out_wiki_winds', \
                    'out_wiki_winds_pca']);'''
for i in dumpPaths:
    printResults(i);
