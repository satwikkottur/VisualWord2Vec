// This is the wrapper around the liblinear library
# include "liblinearWrapper.h"

// Define the problem

// Define the training and test points
struct feature_node** trainNodes, **testNodes;
struct problem* curProblem = NULL; // Current probem to solve using svm
struct parameter* curParam = NULL; // Current parameters for traing svm

// Setting up the whole framework for liblinear SVMs
void setupTrainFrameWork(){
    // Create the problem from the data
    createProblem(trainSents, noTrainVP);

    // Modify the problem structure after getting the new embeddings
    modifyProblem(trainSents, noTrainVP);

    // Setup the parameters for the training
    createParameter();

    // Checking for parameters
    if(check_parameter(curProblem, curParam) != NULL)
        printf("%s\n", check_parameter(curProblem, curParam));
    else
        printf("\nParameters validated!\n");

    // Train the SVM
    train(curProblem, curParam);
    printf("Trained model\n");

    // Cross validate the SVM

}

// Creating the problem structure for liblinear
void createProblem(struct Sentence* trainInsts, long noTrain){
    // Create the struct
    curProblem = (struct problem*) malloc(sizeof(struct problem));
    curProblem->l = (int)noTrain;
    curProblem->n = vpFeatSize; // Dont include the bias
    curProblem->bias = -1;
    curProblem->y = (double*) malloc(sizeof(double) * noTrain);
    curProblem->x = (struct feature_node**) malloc(sizeof(struct feature_node*) * noTrain);

    // Always add a bias
    long i;
    for (i = 0; i < noTrain; i++){
        // store the feature for each point
        curProblem->x[i] = createFeatureNodeList(trainInsts[i], vpFeatSize);
        
        // store the class for each point
        curProblem->y[i] = 1;//(double)(i%2); // Assign random gt
        //curProblem->y[i] = trainInsts[i].gt;
    }
}

// Modify the problem after getting the new embeddings
void modifyProblem(struct Sentence* trainInsts, long noTrain){
    // Assume curProblem is not NULL and modify the features
    long i; int d;
    for (i = 0; i < noTrain; i++)
        for(d = 0; d < vpFeatSize; d++)
            curProblem->x[i][d].value = trainInsts[i].embed[d];
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
}

// Creating the node list for features
struct feature_node* createFeatureNodeList(struct Sentence sent, int featSize){
    // Read the sentence's feature and initialize the list
    struct feature_node* featList = (struct feature_node*) 
                                        malloc(sizeof(struct feature_node) * (featSize+1));
    int d;
    for( d = 0; d < featSize; d++){
        featList[d].index = d + 1;
        featList[d].value = sent.embed[d];
    }
    // Last member should be -1
    featList[featSize].index = -1;

    return featList;
}
