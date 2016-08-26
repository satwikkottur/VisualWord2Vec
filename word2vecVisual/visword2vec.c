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
# include "filepaths.h"
/***********************************************************************************/
// Extern variables
extern float prevTestAcc, prevValAcc;
extern long noTest;

// Variations 
int trainPhrases = 0; // Handle phrases as a unit / separately
int trainMulti = 0; // Train single / multiple models for P,R,S
int numClusters = 25; // Number of initial clusters to use
int windowVP = 5; // window size for training on sentences
// Training the sentences in one of the modes
// Could be one of DESCRIPTIONS, SENTENCES, WORDS, WINDOWS;
enum TrainMode trainMode = SENTENCES;

/***********************************************************************************/
char output_file[MAX_STRING], embed_file[MAX_STRING], mode[MAX_STRING];

long long vocab_size = 0, layer1_size = 100;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0, *syn1;
int num_threads = 1;

int* refineVocab; // Keep track of words that are being refined

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

    printf("\n\n (Phrases, Multi, numClusters) = (%d, %d, %d)\n\n", 
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
        printf("\t\tUse <int> threads (default 1)\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the learning rate; default is 0.01\n");
        printf("\t-clusters <int>\n");
        printf("\t\tNumber of clusters to use; default is 25\n");
        printf("\t-multi <int>\n");
        printf("\t\tTo train single (0) or multiple embeddings(1) (only for cs); default is 0\n");
        printf("\t-phrases <int>\n");
        printf("\t\tHandling phrases together (1) or as separate words (0); default is 0\n");
        printf("\t-cs <int>\n");
        printf("\t\tPerform the cs task (1); default is 0\n");
        printf("\t-vp <int>\n");
        printf("\t\tPerform the vp task (1); default is 0\n");
        printf("\nExamples:\n");
        printf("./word2vec -train data.txt -output vec.txt -size 200 -clusters 30\n\n");
        return 0;
    }

    output_file[0] = 0;

    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-embed-path", argc, argv)) > 0) strcpy(embed_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-clusters", argc, argv)) > 0) numClusters = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-multi", argc, argv)) > 0) trainMulti = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-phrases", argc, argv)) > 0) trainPhrases = atoi(argv[i + 1]);
    int performCS = 0, performVP = 0; // perform either of the task
    if ((i = ArgPos((char *)"-cs", argc, argv)) > 0) performCS = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-vp", argc, argv)) > 0) performVP = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-mode", argc, argv)) > 0) strcpy(mode, argv[i + 1]);
    if ((i = ArgPos((char *)"-window-size", argc, argv)) > 0) windowVP = atoi(argv[i + 1]);

    // Begin the training
    printf("Reading embedding initializations from %s\n", embed_file);
    starting_alpha = alpha;

    // Read the embeddings and initializing the network
    initializeEmbeddings(embed_file);
    
    // Ensure output file is not empty (write something out after done)
    if (output_file[0] == 0) {
        printf("Invalid output file!");
        return -1;
    }

    // Ensure only one task is called atmost
    if (performVP && performCS) {
        printf("Cannot perform both CS and VP!");
        return -1;
    }

    // Visual paraphrase task
    if (performVP){
        if (mode[0] != 0){
            if (strcmp(mode, "DESCRIPTIONS")) trainMode = DESCRIPTIONS;
            if (strcmp(mode, "SENTENCES")) trainMode = SENTENCES;
            if (strcmp(mode, "WORDS")) trainMode = WORDS;
            if (strcmp(mode, "WINDOWS")) trainMode = WINDOWS;
        }
        // Assign the mode
        visualParaphraseWrapper(numClusters);
    }

    // Common sense task
    if (performCS) commonSenseWrapper(numClusters);
    
    // Retriever Wrapper
    //retrieverWrapper();
    
    // Saving the modified embeddings (either separate/shared)
    if (trainMulti) saveWord2VecMulti(output_file);
    else saveWord2VecMulti(output_file);

    return 0;
}
