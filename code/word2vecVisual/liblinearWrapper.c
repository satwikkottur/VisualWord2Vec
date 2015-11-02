// This is the wrapper around the liblinear library
# include "liblinearWrapper.h"

// Define the training and test points
struct feature_node** testNodes; // storing test instances for liblinear
struct feature_node** valNodes; // storing val instances for liblinear
int* testGtruth;
int* valGtruth;

long* trainInds; // Indices of the training instances 
long* testInds; // Indices of the testing instances 
long* valInds; // Indices of the validation instances
long noTrainSVM = 0; // Number of training instances for svm
long noTestSVM = 0; // Number of test instances for svm
long noValSVM = 0; // Number of val instances for svm
long featSizeSVM;
float* testScores = NULL;
float* valScores = NULL;

// A flag to indicate the first run
int firstRun = 0;
int countRun = 0;

struct problem* curProblem = NULL; // Current probem to solve using svm
struct parameter* curParam = NULL; // Current parameters for traing svm
struct model* trainedModel; // Model that is trained

// Setting up the whole framework for liblinear SVMs
void learnClassificationModel(struct SentencePair* sentPairs, long noPairs, int totalFeatSize){
    // Assign the feature dimension (without bias)
    featSizeSVM = totalFeatSize;
    if(noTrainSVM == 0){
        // Classifier and other datastructures uninitialized
        // The first run on the task (record base scores)
        firstRun = 1;
                
        // Populate test and train indices
        populateTestTrainValIndices(sentPairs, noPairs);

        // Create the problem from the data
        createProblemPair(sentPairs);

        // Setup the parameters for the training
        createParameter();

        // Checking for parameters
        if(check_parameter(curProblem, curParam) != NULL)
            printf("%s\n", check_parameter(curProblem, curParam));
        else
            printf("\nParameters validated!\n");

        // Create test nodes
        createTestNodesPair(sentPairs);

        // Create val nodes
        createValNodesPair(sentPairs);
    }
    else{
        // Set the flag to zero (not the first run)
        firstRun = 0;

        // Modify the problem structure after getting the new embeddings
        modifyProblemPair(sentPairs);

        // Modify the test nodes
        modifyTestNodesPair(sentPairs);
        
        // Modify the val nodes
        modifyValNodesPair(sentPairs);
    }
    
    // Find the best parameter C using cross-validation
    double bestAcc, bestC;
    double startC = 0.01;
    double maxC = 10;
    int noFolds = 5;
    
    find_parameter_C(curProblem, curParam, noFolds, startC, maxC, &bestAcc, &bestC);
    printf("\nBest C: %f Best Acc:%f\n\n", bestC, bestAcc);
    
    // Update the parameter with current best, re-train
    curParam->C = bestC;
    // Train the SVM
    trainedModel = train(curProblem, curParam);
    printf("\nTrained best model!\n");

    // Compute the test/val accuracy, using scores from test/val instances
    float mAPTest = computeTestAccuracy(trainedModel, sentPairs);
    float mAPVal = computeValAccuracy(trainedModel, sentPairs);
    printf("\n**************************************\n");
    printf("mAP: Test (%f) Val(%f)\n", mAPTest, mAPVal);
    printf("***************************************\n");
    
    // Also save the 10 sentences from each category based on scores
    char* savePath = (char*) malloc(sizeof(char) * 100);
    if (countRun > 1){
        sprintf(savePath, "modelsNdata/vp_test_scores_%d.txt", countRun);
        saveTestScores(savePath, sentPairs);
    }

    // Free the memory
    //free_and_destroy_model(&trainedModel);
    //free(curParam);
    //free(curProblem);
    //free(testNodes);
    countRun++;
}

// Creating the problem structure for liblinear
void createProblem(struct Sentence* trainInsts, long noTrain){
    // Create the struct
    curProblem = (struct problem*) malloc(sizeof(struct problem));
    curProblem->l = (int)noTrain;
    curProblem->n = layer1_size; // Dont include the bias
    curProblem->bias = -1;
    curProblem->y = (double*) malloc(sizeof(double) * noTrain);
    curProblem->x = (struct feature_node**) malloc(sizeof(struct feature_node*) * noTrain);

    // Always add a bias
    long i;
    for (i = 0; i < noTrain; i++){
        // store the feature for each point
        curProblem->x[i] = createNodeListSentence(trainInsts[i], layer1_size);
        
        // store the class for each point
        //curProblem->y[i] = ((double)i%2) + 1;//(double)(i%2); // Assign random gt
        printf("%ld %ld\n", i%2, i);
        if( i%2 == 0)
            curProblem->y[i] = 1.0;
        else
            curProblem->y[i] = 2.0;
    }
}

// Creating the problem structure for liblinear, given sentence pairs
void createProblemPair(struct SentencePair* sentPairs){
    // Create the struct
    curProblem = (struct problem*) malloc(sizeof(struct problem));
    curProblem->l = (int)noTrainSVM;
    curProblem->n = featSizeSVM; // Dont include the bias
    curProblem->bias = -1;
    curProblem->y = (double*) malloc(sizeof(double) * noTrainSVM);
    curProblem->x = (struct feature_node**) malloc(sizeof(struct feature_node*) * noTrainSVM);

    // Always ignore bias
    long i;
    for (i = 0; i < noTrainSVM; i++){
        // store the feature for each point
        curProblem->x[i] = createNodeListFeature(sentPairs[trainInds[i]].feature);
        
        // store the class for each point
        curProblem->y[i] = sentPairs[trainInds[i]].gt;
    }
}

// Modify the problem after getting the new embeddings
void modifyProblem(struct Sentence* trainInsts, long noTrain){
    // Assume curProblem is not NULL and modify the features
    long i; int d;
    for (i = 0; i < noTrain; i++)
        for(d = 0; d < layer1_size; d++)
            curProblem->x[i][d].value = trainInsts[i].embed[d];
}

// Modify the problem after getting the new embeddings
void modifyProblemPair(struct SentencePair* sentPairs){
    // Assume curProblem is not NULL and modify the features
    // Modify only the first 2*layer1_size
    long i; int d;
    for (i = 0; i < noTrainSVM; i++)
        for(d = 0; d < 2*layer1_size; d++)
            curProblem->x[i][d].value = sentPairs[trainInds[i]].feature[d];
}

// Creating the test nodes using sentence pairs
void createTestNodesPair(struct SentencePair* sentPairs){
    // Initialize the pointers
    testNodes = (struct feature_node**) malloc(sizeof(struct feature_node*) * noTestSVM);
    testGtruth = (int*) malloc(sizeof(int) * noTestSVM);
    
    // Always ignore bias
    long i;
    for (i = 0; i < noTestSVM; i++){
        // store the feature for each point
        testNodes[i] = createNodeListFeature(sentPairs[testInds[i]].feature);
        
        // store the class for each point
        testGtruth[i] = sentPairs[testInds[i]].gt;
    }
}

// Creating the val nodes using sentence pairs
void createValNodesPair(struct SentencePair* sentPairs){
    // Initialize the pointers
    valNodes = (struct feature_node**) malloc(sizeof(struct feature_node*) * noValSVM);
    valGtruth = (int*) malloc(sizeof(int) * noValSVM);
    
    // Always ignore bias
    long i;
    for (i = 0; i < noValSVM; i++){
        // store the feature for each point
        valNodes[i] = createNodeListFeature(sentPairs[valInds[i]].feature);
        
        // store the class for each point
        valGtruth[i] = sentPairs[valInds[i]].gt;
    }
}

// Creating generic feature nodes (doesn't work)
void createFeatureNodesPair(struct SentencePair* sentPairs, 
                            struct feature_node** nodes,
                            int* gTruth,
                            long* inds, long count){
    // initialize the pointers
    nodes = (struct feature_node**) malloc(sizeof(struct feature_node*) * count);
    gTruth = (int*) malloc(sizeof(int) * count);

    // Always ignore bias
    long i;
    for (i = 0; i < count; i++){
        // store the feature for each point
        nodes[i] = createNodeListFeature(sentPairs[inds[i]].feature);
        
        // store the class for each point
        gTruth[i] = sentPairs[inds[i]].gt;
    }
}

// Modifying the test nodes using sentence pairs
void modifyTestNodesPair(struct SentencePair* sentPairs){
    if(testNodes == NULL){
        printf("\nTest nodes empty! Cannot modify!\n");
        exit(1);
    }

    // Assume testNodes is not NULL and modify the features
    // Modify only the first 2*layer1_size
    long i; int d;
    for (i = 0; i < noTestSVM; i++)
        for(d = 0; d < 2*layer1_size; d++)
            testNodes[i][d].value = sentPairs[testInds[i]].feature[d];
}

// Modifying the val nodes using sentence pairs
void modifyValNodesPair(struct SentencePair* sentPairs){
    if(valNodes == NULL){
        printf("\nVal nodes empty! Cannot modify!\n");
        exit(1);
    }

    // Assume valNodes is not NULL and modify the features
    // Modify only the first 2*layer1_size
    long i; int d;
    for (i = 0; i < noValSVM; i++)
        for(d = 0; d < 2*layer1_size; d++)
            valNodes[i][d].value = sentPairs[valInds[i]].feature[d];
}

// Modifying generic feature nodes - test / val
void modifyFeatureNodesPair(struct SentencePair* sentPairs,
                            struct feature_node** nodes,
                            int* gTruth,
                            long* inds, long count){
                            
                            
}

// Creating the problem parameters
void createParameter(){
    // Solver type
    curParam = (struct parameter*) malloc(sizeof(struct parameter));

    curParam->solver_type = L2R_L2LOSS_SVC;
    curParam->C = 0.1;
    curParam->eps = 0.01; // default
    curParam->p = 0.1; //default
    curParam->nr_weight = 0;
    curParam->init_sol = NULL;
}

// Creating the node list for features
struct feature_node* createNodeListSentence(struct Sentence sent, int featSize){
    // Read the sentence's feature and initialize the list
    struct feature_node* featList = (struct feature_node*) 
                                        malloc(sizeof(struct feature_node) * (featSize+1));
    int d;
    for(d = 0; d < featSize; d++){
        featList[d].index = d + 1;
        featList[d].value = sent.embed[d];
    }
    // Last member should be -1
    featList[featSize].index = -1;

    return featList;
}

// Creating the node list for features
struct feature_node* createNodeListFeature(float* feature){
    // TODO: Use only relevant amount of featList
    // Read the sentence's feature and initialize the list
    struct feature_node* featList = (struct feature_node*) 
                                        malloc(sizeof(struct feature_node) * (featSizeSVM+1));
    int d, featCount = 0; 
    for(d = 0; d < featSizeSVM; d++){
        // First 2 * layer1_size should always be dense, and then sparsify
        if(d < 2*layer1_size){
            featList[d].index = d + 1;
            featList[d].value = feature[d];
            featCount++;
        }
        else{
            // We need sparse representation, ignore small values
            if(feature[d] > 1e-5){
                featList[featCount].index = d + 1;
                featList[featCount].value = feature[d];
                featCount++;
            }
        }
    }
    // Last member should be -1
    featList[featCount].index = -1;

    return featList;
}

// Populate the test and train indices from the sentence pairs
void populateTestTrainValIndices(struct SentencePair* sentPairs, long noPairs){
    // First go through the pairs and determine the count of test/train
    long i;
    // Re-set number of train and test
    noTrainSVM = 0;
    noTestSVM = 0;
    noValSVM = 0;

    for(i = 0; i < noPairs; i++)
        if(sentPairs[i].isVal)
            // Val set
            noValSVM++;
        else if(sentPairs[i].isTrain)
                // Train
                noTrainSVM++;
            else
                // Test
                noTestSVM++;

    // Now populate the train and test indices
    trainInds = (long*) malloc(sizeof(long) * noTrainSVM);
    testInds = (long*) malloc(sizeof(long) * noTestSVM);
    valInds = (long*) malloc(sizeof(long) * noValSVM);
    long trainCount = 0, testCount = 0, valCount = 0;
    for(i = 0; i < noPairs; i++){
        // First check if its val set
        if(sentPairs[i].isVal){
            // Val
            valInds[valCount] = i;
            valCount ++;
        }
        else if(sentPairs[i].isTrain){
            // Train
            trainInds[trainCount] = i;
            trainCount++;
        }
        else{
            // Test
            testInds[testCount] = i;
            testCount++;
        }
    }

    printf("Total : Train (%ld) Test (%ld) Val (%ld)\n", noTrainSVM, noTestSVM, noValSVM);
}

// Computing the accuracy for the test set, given the model
float computeTestAccuracy(struct model* trainedModel, struct SentencePair* sentPairs){
    // Initialize score array, if NULL
    if(testScores == NULL) testScores = (float*) malloc(sizeof(float) * noTestSVM);

    double score;
    // Loop through all the instances and predict
    long i;
    for(i = 0; i < noTestSVM; i++){
        // Compute the score and store
        predict_values(trainedModel, testNodes[i], &score);
        testScores[i] = score;
        
        // Now record the test scores in the struct to get the best improved sentences
        if(firstRun) sentPairs[testInds[i]].baseScore = score;
        else sentPairs[testInds[i]].newScore = score;
    }

    // Now pedict the mAP, using ranks
    printf("Computing MAP...\n\n");
    // Crude implementation
    long a, b;
    int* rankedLabels = (int*) malloc(sizeof(int) * noTestSVM);
    float* rankedScores = (float*) malloc(sizeof(float) * noTestSVM);

    // Make a copy of scores
    for(a = 0; a < noTestSVM; a++) rankedScores[a] = testScores[a];

    long maxInd;
    // Get the rank of positive instances wrt to the scores
    for(a = 0; a < noTestSVM; a++){
        maxInd = 0;
        for(b = 0; b < noTestSVM; b++)
            // Check if max is also current max (flag out using -5)
            if(rankedScores[b] != 1e-10 && 
                        rankedScores[maxInd] < rankedScores[b]) 
                maxInd = b;

        // Swap the max and element at that instance
        rankedLabels[a] = testGtruth[maxInd];
        // NULLing the max ind 
        rankedScores[maxInd] = -5;
    }

    float mAP = 0;
    long noPositives = 0;
    // Compute the similarity wrt ideal ordering 1,2,3.....
    for(a = 0; a < noTestSVM; a++)
        if(rankedLabels[a] == 1){
            // Increasing the positives
            noPositives++;
            mAP += noPositives / (float) (a + 1);
            //printf("%ld %ld %f\n", noPositives, a + 1, noPositives/(float) (a + 1));
        }
   
    // Compute mAP
    mAP = mAP / noPositives;

    // Free memory
    free(rankedScores);
    free(rankedLabels);
    return mAP;
}

// Computing the accuracy for the val set, given the model
float computeValAccuracy(struct model* trainedModel, struct SentencePair* sentPairs){
    // Initialize score array, if NULL
    if(valScores == NULL) valScores = (float*) malloc(sizeof(float) * noValSVM);

    double score;
    // Loop through all the instances and predict
    long i;
    for(i = 0; i < noValSVM; i++){
        // Compute the score and store
        predict_values(trainedModel, valNodes[i], &score);
        valScores[i] = score;
    }

    // Now pedict the mAP, using ranks
    printf("Computing MAP...\n\n");
    // Crude implementation
    long a, b;
    int* rankedLabels = (int*) malloc(sizeof(int) * noValSVM);
    float* rankedScores = (float*) malloc(sizeof(float) * noValSVM);

    // Make a copy of scores
    for(a = 0; a < noValSVM; a++) rankedScores[a] = valScores[a];

    long maxInd;
    // Get the rank of positive instances wrt to the scores
    for(a = 0; a < noValSVM; a++){
        maxInd = 0;
        for(b = 0; b < noValSVM; b++)
            // Check if max is also current max (flag out using -5)
            if(rankedScores[b] != 1e-10 && 
                        rankedScores[maxInd] < rankedScores[b]) 
                maxInd = b;

        // Swap the max and element at that instance
        rankedLabels[a] = valGtruth[maxInd];
        // NULLing the max ind 
        rankedScores[maxInd] = -5;
    }

    float mAP = 0;
    long noPositives = 0;
    // Compute the similarity wrt ideal ordering 1,2,3.....
    for(a = 0; a < noValSVM; a++)
        if(rankedLabels[a] == 1){
            // Increasing the positives
            noPositives++;
            mAP += noPositives / (float) (a + 1);
            //printf("%ld %ld %f\n", noPositives, a + 1, noPositives/(float) (a + 1));
        }
   
    // Compute mAP
    mAP = mAP / noPositives;

    // Free memory
    free(rankedScores);
    free(rankedLabels);
    return mAP;
}

// Save the test scores along with ground truth and id
void saveTestScores(char* fileName, struct SentencePair* sentPairs){
    FILE* filePt = fopen(fileName, "wb");
    
    int i;
    for(i = 0; i < noTestSVM; i++){
        fprintf(filePt, "%d %d %f %f\n", testInds[i], 
                                    sentPairs[testInds[i]].gt, 
                                    sentPairs[testInds[i]].baseScore,
                                    sentPairs[testInds[i]].newScore);
    }

    fclose(filePt);
}

    /*int i, j;
    for (i = 0; i < noTrainVP; i++){
        for(j = 0; j < layer1_size + 1; j++){
            printf("(%f %d ; ", curProblem->x[i][j].value, curProblem->x[i][j].index);
            printf("%f )\n", curProblem->y[i]);
        }
    }
    printf("l:%d n:%d\n", curProblem->l, curProblem->n);
    // Cross validation
    double* values = (double*) malloc(sizeof(double) * noTrainVP * noFolds);
    int i = 0, j;
    for(j = 0; j < noFolds; j++)
        for (i = 0; i < noTrainVP; i++){
            values[noTrainVP * j + i] = -10;
    }
    cross_validation(curProblem, curParam, noFolds, values);*/
