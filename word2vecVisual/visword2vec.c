//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <ctype.h>
#include <unistd.h>

/**********************************************************************************/
// [S] added
# include "macros.h"
# include "structs.h"
# include "visualFeatures.h"
# include "vpFunctions.h"
# include "helperFunctions.h"
# include "retriever.h"
# include "filepaths.h"
/***********************************************************************************/
// Extern variables
extern float prevTestAcc, prevValAcc;
extern long noTest;

extern float *syn0P, *syn0S, *syn0R;

// Variations 
int usePCA = 0; // Using PCA
int trainPhrases = 0; // Handle phrases as a unit / separately
int trainMulti = 0; // Train single / multiple models for P,R,S
int clusterCommonSense = 25; // Number of initial clusters to use
int clusterVP = 100; // Number of initial clusters to use
int permuteMAP = 0; // Permute the data and compute mAP multiple times
int debugModeVP = 0; // Debug mode for VP task
int debugModeVQA = 0; // Debug mode for VQA task
int windowVP = 5; // window size for training on sentences
// Training the sentences in one of the modes
// Could be one of DESCRIPTIONS, SENTENCES, WORDS, WINDOWS;
enum TrainMode trainMode = SENTENCES;

/***********************************************************************************/
const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

char train_file[MAX_STRING], output_file[MAX_STRING];
char embed_file[MAX_STRING];
struct vocab_word *vocab;
int num_threads = 12;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0, *syn1;
clock_t start;

int* refineVocab; // Keep track of words that are being refined

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Function for common sense task
void commonSenseWrapper(int clusterArg){
    // Common sense task
    // Initializing the hash, for storing the tuples
    initFeatureHash();

    // Reading for the word features, cluster ids and visual features
    // Use the second argument to give prs tuple file different from train of CS
    // Else NULL
    readRefineTrainFeatureFiles(CS_PRS_TRAIN_FILE, NULL);
    
    // Clustering in C
    noClusters = 0;
    readVisualFeatureFile(CS_VISUAL_FEATURE_FILE);
   
    // To save clusterId / distance, provide save path in the second part; else NULL
    clusterVisualFeatures(clusterArg, NULL);
    
    // Read the validation and test sets    
    if(noTest == 0)
        // Read the strings for test and validation sets, store features
        readTestValFiles(CS_PRS_VAL_FILE, CS_PRS_TEST_FILE);

    // Store the basemodel test tuple scores and best model test tuple scores
    float* baseTestScores = (float*) malloc(sizeof(float) * noTest);
    float* bestTestScores = (float*) malloc(sizeof(float) * noTest);

    // Perform common sense task before refining
    if(trainMulti){
        // Initializing the refining network
        initMultiRefining();
        // Performing the multi model common sense task
        performMultiCommonSenseTask(baseTestScores);
    }
    else{
        // Initializing the refining network
        initRefining();
        // Perform common sense task
        performCommonSenseTask(baseTestScores);
    }

    // Reset valAccuracy as the first run doesnt count
    prevValAcc = 0; 
    prevTestAcc = 0;

    printf("\n\n (phrases, multi, noClusters) = (%d, %d, %d)\n\n", 
                                        trainPhrases, trainMulti, clusterArg);
    
    int noOverfit = 1;
    // Keep refining until val performance saturates
    while(noOverfit){
        if(trainMulti){
            // Refine the network for multi model, based on words or phrases
            if(trainPhrases) refineMultiNetworkPhrase();
            else refineMultiNetwork();

            // Performing the multi model common sense task
            noOverfit = performMultiCommonSenseTask(bestTestScores);
        }
        else{
            // Refine the network, based on words or phrases
            if(trainPhrases) refineNetworkPhrase();
            else refineNetwork();

            // Perform common sense task
            noOverfit = performCommonSenseTask(bestTestScores);
        }
    }
}

// Function for text retriever task
void retrieverWrapper(){
    // Rename clusterArg in current function
    int clusterArg = clusterCommonSense;

    // Load the word2vec embeddings from Xiao's
    //char wordPath[] = "/home/satwik/VisualWord2Vec/word2vecVisual/modelsNdata/al_vectors.txt";
    //char wordPath[] = "modelsNdata/vis-genome/word2vec_genome_train.bin";
    //char wordPath[] = "/home/satwik/VisualWord2Vec/models/wiki_embeddings.bin";
    char wordPath[] = "/home/satwik/VisualWord2Vec/data/coco-cnn/word2vec_coco_caption_before.bin";
    //char wordPath[] = "modelsNdata/vis-genome/word2vec_genome_02.bin";
    loadWord2Vec(wordPath);

    // [S] added
    char* visualPath = (char*) malloc(sizeof(char) * 100);
    char* postPath = (char*) malloc(sizeof(char) * 100);
    char* prePath = (char*) malloc(sizeof(char) * 100);
    char* vocabPath = (char*) malloc(sizeof(char) * 100);
    char* embedDumpPath = (char*) malloc(sizeof(char) * 100);
    char* featurePathICCV = (char*) malloc(sizeof(char) * 100);
    char* featurePathCOCO = (char*) malloc(sizeof(char) * 100);
    char* featurePathVQA = (char*) malloc(sizeof(char) * 100);
    char* testFile = (char*) malloc(sizeof(char) * 100);
    char* valFile = (char*) malloc(sizeof(char) * 100);

    // Common sense task
    // Reading the file for relation word
    //featurePathVQA = "/home/satwik/VisualWord2Vec/data/vqa/vqa_psr_features.txt";
    //featurePathCOCO = "/home/satwik/VisualWord2Vec/data/coco-cnn/PRS_features_coco.txt";
    featurePathICCV = "/home/satwik/VisualWord2Vec/data/PRS_features.txt";
    //char featurePath[] = "/home/satwik/VisualWord2Vec/data/PSR_features.txt";
    //char featurePath[] = "/home/satwik/VisualWord2Vec/data/PSR_features_lemma.txt";
    //char featurePath[] = "/home/satwik/VisualWord2Vec/data/PSR_features_18.txt";
    //char featurePath[] = "/home/satwik/VisualWord2Vec/data/PSR_features_R_120.txt";

    //char featurePath[] = "/home/satwik/VisualWord2Vec/data/vp_train_sentences_lemma.txt";

    //char clusterPath[] = "/home/satwik/VisualWord2Vec/code/clustering/clusters_10.txt";
    sprintf(postPath, "/home/satwik/VisualWord2Vec/word2vecVisual/modelsNdata/word2vec_wiki_post_%d_%d_%d_%d.txt", 
                                        trainPhrases, usePCA, trainMulti, clusterArg);
    sprintf(prePath, "/home/satwik/VisualWord2Vec/word2vecVisual/modelsNdata/word2vec_wiki_pre_%d_%d_%d_%d.txt", 
                                        trainPhrases, usePCA, trainMulti, clusterArg);
    sprintf(vocabPath, "/home/satwik/VisualWord2Vec/word2vecVisual/modelsNdata/word2vec_vocab_%d_%d_%d_%d.txt",
                                        trainPhrases, usePCA, trainMulti, clusterArg);
    testFile = "/home/satwik/VisualWord2Vec/data/test_features.txt";
    valFile = "/home/satwik/VisualWord2Vec/data/val_features.txt";

    //visualPath = "/home/satwik/VisualWord2Vec/data/float_features_18.txt";
    //visualPath = "/home/satwik/VisualWord2Vec/data/coco-cnn/float_features_coco.txt";
    //visualPath = "/home/satwik/VisualWord2Vec/data/vqa/vqa_float_features.txt";
    visualPath = "/home/satwik/VisualWord2Vec/data/float_features.txt";
    //visualPath = "/home/satwik/VisualWord2Vec/data/float_features_R_120.txt";

    // Writing word2vec from file
    //char wordPath[] = "/home/satwik/VisualWord2Vec/code/word2vecVisual/modelsNdata/word2vec_save.txt";
    //saveWord2Vec(wordPath);

    // Initializing the hash
    initFeatureHash();
    // Reading for the word features, cluster ids and visual features
    // clusterid reading will be avoided when clustering is ported to c
    readRefineTrainFeatureFiles(featurePathICCV, NULL);
    
    // reading cluster files from matlab
    //char clusterpath[] = "/home/satwik/visualword2vec/data/coco-cnn/cluster_100_coco_train.txt";
    //readclusteridfile(clusterpath);
    // Clustering in C
    noClusters = 0;
    readVisualFeatureFile(visualPath);
    char clusterSavePath[] = "/home/satwik/VisualWord2Vec/word2vecVisual/modelsNdata/cluster_id_save.txt";
    // To save clusterId / distance, provide save path; else NULL
    clusterVisualFeatures(clusterArg, NULL);
    //gmmVisualFeatures(clusterArg, NULL);
    //return;
    
    // Read the validation and test sets    
    if(noTest == 0)
        // Clean the strings for test and validation sets, store features
        readTestValFiles(valFile, testFile);

    // Saving the feature word vocabulary(split simply means the corresponding components)
    //saveFeatureWordVocab(vocabPath);
    //char splitPath[] = "/home/satwik/VisualWord2Vec/code/word2vecVisual/modelsNdata/split_vocab.txt";  
    //saveFeatureWordVocabSplit(splitPath);
    // Saving the feature vocabulary
    //saveFeatureWordVocab(vocabPath);
    
    // Store the basemodel test tuple scores and best model test tuple scores
    float* baseTestScores = (float*) malloc(sizeof(float) * noTest);
    float* bestTestScores = (float*) malloc(sizeof(float) * noTest);

    if(trainMulti){
        // Initializing the refining network
        initMultiRefining();
        // Performing the multi model common sense task
        performMultiCommonSenseTask(baseTestScores);
    }
    else{
        // Initializing the refining network
        initRefining();
        // Perform common sense task
        performCommonSenseTask(baseTestScores);
    }

    // Saving the embeddings, before refining
    /*if(trainMulti)
        saveMultiEmbeddings(prePath);
    else
        saveEmbeddings(prePath);*/

    // Reset valAccuracy as the first run doesnt count
    prevValAcc = 0; 
    prevTestAcc = 0;

    printf("\n\n (PCA, phrases, multi, noClusters) = (%d, %d, %d, %d)\n\n", 
                                        usePCA, trainPhrases, trainMulti, clusterArg);
    
    int noOverfit = 1;
    int iter = 0;

    // Read the train and test retriever
    char rValPath[] = "/home/satwik/VisualWord2Vec/data/coco-cnn/captions_coco_val_nomaps.txt";
    char rGtValPath[] = "/home/satwik/VisualWord2Vec/data/coco-cnn/captions_coco_val_gtruth.txt";
    char rTestPath[] = "/home/satwik/VisualWord2Vec/data/coco-cnn/captions_coco_test_nomaps.txt";
    char rGtTestPath[] = "/home/satwik/VisualWord2Vec/data/coco-cnn/captions_coco_test_gtruth.txt";
    char rTrainPath[] = "/home/satwik/VisualWord2Vec/data/coco-cnn/captions_coco_dataset_nomaps.txt";
    // Read all the training, validation sentences and map
    readTestValRetriever(rTrainPath, rValPath, rGtValPath, rTestPath, rGtTestPath);

    // Perform the retrieval task
    performRetrieval();

    int i, noIters = 10;
    for(i = 0; i < noIters; i++){
        printf("Refining : %d / %d\n", i, noIters);

        // Refining the embeddings
        refineNetwork();
        
        // Perform the retrieval task
        performRetrieval();
    }

    /*while(noOverfit){
        // Refine the network for multi model
        if(trainMulti){
            if(trainPhrases)
                refineMultiNetworkPhrase();
            else
                refineMultiNetwork();
        }
        // Refine the network
        else{
            if(trainPhrases)
                refineNetworkPhrase();
            else
                refineNetwork();
        }

        // Saving the embeddings snapshots
        //sprintf(embedDumpPath, "/home/satwik/VisualWord2Vec/code/word2vecVisual/modelsNdata/word2vec_wiki_iter_%d.bin",
        //                                    iter);
        //saveWord2Vec(embedDumpPath);
        //iter++;
        
        if(trainMulti)
            // Performing the multi model common sense task
            //noOverfit = performMultiCommonSenseTask(NULL);
            noOverfit = performMultiCommonSenseTask(bestTestScores);
        else
            // Perform common sense task
            //noOverfit = performCommonSenseTask(NULL);
            noOverfit = performCommonSenseTask(bestTestScores);
    }

    // Saving the embeddings, after refining
    if(trainMulti)
        saveMultiEmbeddings(postPath);
    else
        saveEmbeddings(postPath);*/

    // Find test tuples with best improvement, for further visualization
    //findBestTestTuple(baseTestScores, bestTestScores);

    // Read and perform common sense testing on different sets
    /*int fileId = 0;
    for (fileId = 0; fileId < 20; fileId++){
        printf("Test case: %d...\n", fileId);
        testFile = (char*) malloc(100 * sizeof(char));
        sprintf(testFile, "/home/satwik/VisualWord2Vec/data/common-sense/test_features_subset_%02d.txt",
                                                            fileId);
        // Perform the common sense task on the current subset;
        readTestValFiles(valFile, testFile);
        if (trainMulti)
            noOverfit = performMultiCommonSenseTask(bestTestScores);
        else
            noOverfit = performCommonSenseTask(bestTestScores);
    }*/
}

// Function for visual paraphrase task
void visualParaphraseWrapper(int clusterArg){
    // Reading the file for training
    char* visualPath = (char*) malloc(sizeof(char) * 100);
    char* featurePath = (char*) malloc(sizeof(char) * 100);

    // Reading for the word features and visual features
    readVPTrainSentences(VP_TRAIN_CAPTION_FILE);
    readVPAbstractVisualFeatures(VP_VISUAL_FEATURE_FILE);
    
    // Tokenizing the training sentences
    tokenizeTrainSentences();
    
    // Perform the VP task (text only)
    performVPTask();

    // Clustering the visual features
    clusterVPAbstractVisualFeatures(clusterArg, NULL);

    // Begin the refining (run for 100 iterations and choose the best based on 
    // val performance)
    // Initializing the refining network
    initRefining();

    // Save path for the word2vec
    int i, noIters = 100;
    for(i = 0; i < noIters; i++){
        printf("Refining : %d / %d\n", i, noIters);

        // Refining the embeddings
        refineNetworkVP();
        
        // Compute the vp task
        performVPTask();
    }
}

// Initializing the network and read the embeddings
void initializeNetwork(char* embedPath){
    FILE* filePt = fopen(embedPath, "rb");
    
    long long i, j, offset;
    long dims;
    float value;
    char word[MAX_STRING];
    unsigned int hash, length;

    //------------------------------------------------------------
    // Read vocab size and number of hidden layers
    if (!fscanf(filePt, "%lld %ld\n", &vocab_size, &dims)){
        printf("Error reading the embed file!");
        exit(1);
    }

    if(layer1_size != dims){
        printf("Number of dimensions not consistent with embedding file!\n");
        exit(1);
    }
    //------------------------------------------------------------
    // Allocate memory for the intput-to-hidden parameters
    printf("(vocab size, hidden dims): %lld %lld\n", vocab_size, layer1_size);
    int flag = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn0 == NULL || flag){
        printf("Memory allocation failed\n");
        exit(1);
    }
    //------------------------------------------------------------
    // Allocating memory for reading vocab, initialize hash
    vocab = (struct vocab_word*) malloc(vocab_size * sizeof(struct vocab_word));
    vocab_hash = (int*) malloc(sizeof(int) * vocab_hash_size);
    for (i = 0; i < vocab_hash_size; i++) vocab_hash[i] = -1;

    // Initializing with random values
    //unsigned long long next_random = 1;

    // Reading the words and feature and store them sequentially
    for (i = 0; i < vocab_size; i++){
        // Store the word
        if (!fscanf(filePt, "%s", word)){
            printf("Error reading the embed file!");
            exit(1);
        }
        //printf("%lld, %lld, %s\n", i, vocab_size, word);
        
        length = strlen(word) + 1;
        // Truncate if needed
        if (length > MAX_STRING) length = MAX_STRING;

        vocab[i].word = (char*) calloc(length, sizeof(char));
        strcpy(vocab[i].word, word);
        vocab[i].cn = 0;

        hash = GetWordHash(word);
        while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
        vocab_hash[hash] = i;
        // Store feature
        offset = layer1_size * i;
        for (j = 0; j < layer1_size; j++){
            if (!fscanf(filePt, "%f", &value)){
                printf("Error reading the embed file!");
                exit(1);
            }

            // Initializing with random
            //next_random = next_random * (unsigned long long)25214903917 + 11;
            //syn0[offset + j] = (((next_random & 0xffff) / (real)65536) - 0.5) / layer1_size;

            // Storing the value
            syn0[offset + j] = value;
        }
    }

    // Close the file and exit
    fclose(filePt);
    printf("Done reading and initializing embeddings...\n");
}

// Bulk of the work is done here
void trainModel() {
    printf("Reading embedding initializations from %s\n", embed_file);
    starting_alpha = alpha;

    // Read the embeddings and initializing the network
    initializeNetwork(embed_file);
    
    // Ensure output file is not empty (write something out after done)
    if (output_file[0] == 0) return;

    // Visual paraphrase task
    //visualParaphraseWrapper(clusterVP);

    // Common sense task
    // Pass in the number of clusters to use for refining
    commonSenseWrapper(clusterCommonSense);
    //return;
    
    // Retriever Wrapper
    //retrieverWrapper();
    return;

    //******************************************************************
    /**************************************************************/
    // skip writing to the file
    /***************************************************************/
    // Write the three models separately (P,R,S)
    // P 
    /*char outputP[] = "/home/satwik/VisualWord2Vec/models/p_wiki_model.txt";
    fo = fopen(outputP, "wb");
    syn0 = syn0P;
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
    fclose(fo);
    
    // R
    char outputR[] = "/home/satwik/VisualWord2Vec/models/r_wiki_model.txt";
    fo = fopen(outputR, "wb");
    syn0 = syn0R;
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
    fclose(fo);

    // S
    char outputS[] = "/home/satwik/VisualWord2Vec/models/s_wiki_model.txt";
    fo = fopen(outputS, "wb");
    syn0 = syn0S;
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
    fclose(fo);
    return;*/
    //=================================================
    // Code to save embeddings
    long a, b;
    FILE* fo = fopen(output_file, "wb");
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
    fclose(fo);
}

// Obtain the position of an argument in the list
int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

// Read the arguments and setup corrresponding flags / variables
int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("Visual Word2Vec:\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-embed-path <file>\n");
        printf("\t\tPath to pre-trained embeddings to use for refining\n");
        //printf("\t-train <file>\n");
        //printf("\t\tUse text data from <file> to train the model\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the resulting word vectors\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of word vectors; default is 100\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 12)\n");
        printf("\t-iter <int>\n");
        printf("\t\tRun more training iterations (default 5)\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
        printf("\nExamples:\n");
        printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
        return 0;
    }

    output_file[0] = 0;

    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
    //if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-embed-path", argc, argv)) > 0) strcpy(embed_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);

    vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));

    // Begin the training
    trainModel();
    return 0;
}
