/* 
 *  Code to train visual word2vec
 *  Link: https://github.com/satwikkottur/VisualWord2Vec
 *  Author: Satwik Kottur
 *  Email: skottur@andrew.cmu.edu
 *
 *  Code inspired from: Google word2vec C code
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <ctype.h>
#include <unistd.h>

/**********************************************************************************/
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
int trainPhrases = 0; // Handle phrases as a unit / separately
int trainMulti = 0; // Train single / multiple models for P,R,S
int numClusters = 25; // Number of initial clusters to use
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

// Function for visual paraphrase task
void visualParaphraseWrapper(int clusterArg){
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
    commonSenseWrapper(numClusters);
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
        printf("\t-alpha <float>\n");
        printf("\t\tSet the learning rate; default is 0.01\n");
        printf("\t-clusters <int>\n");
        printf("\t\tNumber of clusters to use; default is 25\n");
        printf("\t-multi <int>\n");
        printf("\t\tTo train single (0) or multiple embeddings(1) (only for cs); default is 0\n");
        printf("\t-phrases <int>\n");
        printf("\t\tHandling phrases together (1) or as separate words (0); default is 0\n");
        printf("\nExamples:\n");
        printf("./word2vec -train data.txt -output vec.txt -size 200 -clusters 30\n\n");
        return 0;
    }

    output_file[0] = 0;

    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
    //if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-embed-path", argc, argv)) > 0) strcpy(embed_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-clusters", argc, argv)) > 0) numClusters = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-clusters", argc, argv)) > 0) trainMulti = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-clusters", argc, argv)) > 0) trainPhrases = atoi(argv[i + 1]);

    vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));

    // Begin the training
    trainModel();
    return 0;
}
