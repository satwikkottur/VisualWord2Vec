// This is the wrapper around the liblinear library
# include "liblinearWrapper.h"

// Define the training and test points
struct feature_node** testNodes; // storing test instances for liblinear
int* gtTest; // Ground truth for the test instances
long noTestInst; // Number of test instances
struct problem* curProblem = NULL; // Current probem to solve using svm
struct parameter* curParam = NULL; // Current parameters for traing svm

// Setting up the whole framework for liblinear SVMs
void learnClassificationModel(){
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
    struct model* trainedModel = train(curProblem, curParam);
    printf("Trained model\n");
    
    // Find the best parameter C using cross-validation
    double bestAcc, bestC;
    double startC = 0.001;
    double maxC = 100;
    int noFolds = 5;
    
    find_parameter_C(curProblem, curParam, noFolds, startC, maxC, &bestAcc, &bestC);
    printf("Best C: %f\nBest Acc:%f\n", bestC, bestAcc);

    // Cross validation
    /*double* values = (double*) malloc(sizeof(double) * noTrainVP * noFolds);
    int i = 0, j;
    for(j = 0; j < noFolds; j++)
        for (i = 0; i < noTrainVP; i++){
            values[noTrainVP * j + i] = -10;
    }
    cross_validation(curProblem, curParam, noFolds, values);*/

    // Test for instances
    // float accuracy = computeAccuracy(trainedModel, curProblem->x, curProblem->y, curProblem->l);
    // float accuracy = computeAccuracy(trainedModel, testNodes, gtTest, noTestInst);
    
    // Free the memory
    free_and_destroy_model(&trainedModel);
    free(curParam);
    free(curProblem);
    //free(testNodes);
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
        curProblem->x[i] = createFeatureNodeList(trainInsts[i], layer1_size);
        
        // store the class for each point
        //curProblem->y[i] = ((double)i%2) + 1;//(double)(i%2); // Assign random gt
        printf("%ld %ld\n", i%2, i);
        if( i%2 == 0)
            curProblem->y[i] = 1.0;
        else
            curProblem->y[i] = 2.0;
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
struct feature_node* createFeatureNodeList(struct Sentence sent, int featSize){
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

// Computing the accuracy for the test set, given the model
float computeAccuracy(struct model* trainedModel, struct feature_node** insts, int* gt, long noInsts){
    // Initialize accuracy
    //float accuracy;
    int noCorrect = 0;
    float predictClass;
    double score;
    // Loop through all the instances and predict
    long i;
    for(i = 0; i < noInsts; i++){
        // Predict the class
        predict_values(trainedModel, insts[i], &score);
        printf("%f\n", score);

        // Compute accuracy
        //if(gt[i] == predictClass) noCorrect++;
    }

    //return (float)noCorrect/noInsts;
    return 1.0;
}

    /*int i, j;
    for (i = 0; i < noTrainVP; i++){
        for(j = 0; j < layer1_size + 1; j++){
            printf("(%f %d ; ", curProblem->x[i][j].value, curProblem->x[i][j].index);
            printf("%f )\n", curProblem->y[i]);
        }
    }
    printf("l:%d n:%d\n", curProblem->l, curProblem->n);*/
