// Definitions for functions related to the visual paraphrasing task
# include "vpFunctions.h"

// Globals for the current scope
long noTrainVP = 0;
int vpFeatSize = 0;
struct trainSent* train;

// Read the training sentences for VP task
void readVPTrainSentences(char* featurePath){
    // Open the file
    FILE* filePtr = fopen(featurePath, "rb");

    if(filePtr == NULL){
        printf("File not found !\n");
        exit(1);
    }

    // Buffer to store the current sentence
    char* currentSent = (char*) malloc(sizeof(char) * MAX_SENTENCE);

    long sentCount = 0, i;
    // Read lines one by one, just counting number of sentences
    while(fgets(currentSent, MAX_SENTENCE, filePtr) != NULL)
        sentCount++;

    // Rewind the stream and read again
    rewind(filePtr);
    
    // Allocate the train
    train = (struct trainSent*) malloc(sizeof(struct trainSent) * sentCount);
    // Read and store contents
    for( i = 0; i < sentCount; i++){
        // Allocating other memories
        //train[i].count = 0;
        //train[i].index = (int*) malloc(0);

        if(fgets(currentSent, MAX_SENTENCE, filePtr) != NULL){
            // Remove the trailing \n
            currentSent[strlen(currentSent) - 1] = '\0';
            
            // Allocate memory and copy sentence
            train[i].sent = (char*) malloc(sizeof(char) * MAX_SENTENCE);
            strcpy(train[i].sent, currentSent);
            
        }
    }

    // Checking with noTrainVP, if exists, else initializing
    if(noTrainVP != 0){
        if(noTrainVP != sentCount){
            printf("Mismatch with number of training examples: %s\n", featurePath);
            exit(1);
        }
    }
    else noTrainVP = sentCount;

    fclose(filePtr);
    printf("\nFile read with %ld sentences!....\n", sentCount);
    
    ////////////////////////////////////////////////////////////////////
    // Writing it back to the file for debugging
    /*FILE* savePtr = fopen("vp_train_emit.txt", "wb");

    for (i = 0; i < noTrainVP; i++){
        fprintf(savePtr, "%s\n", train[i].sent);
    }

    fclose(savePtr);*/
}

// Read the visual features for VP task
void readVPVisualFeatures(char* visualPath){
    FILE* filePt = fopen(visualPath, "rb");

    if(filePt == NULL){
        printf("File not found for visual features!\n");
        exit(1);
    }

    // Read first line the dimension of visual features
    int featDim = 0;
    fscanf(filePt, "%d\n", &featDim);
    printf("\nVisual feature size : %d\n", featDim);
    
    int noLines = 0, i;
    float feature;
    while(fscanf(filePt, "%f", &feature) != EOF){
        // Allocate memory
        train[noLines].vFeat = (float*) malloc(sizeof(float) * featDim);
       
        // First entry
        train[noLines].vFeat[0] = feature;

        for (i = 1; i < featDim; i++){
            fscanf(filePt, "%f", &feature);
            train[noLines].vFeat[i] = feature;
        }
    
        noLines++;
    }

    // Tallying with existing numbers
    if(noTrainVP != 0 && noTrainVP != noLines){
        printf("\nNumber of training sentences and visual features dont match!\n");
        printf("NoTrainVP: %d\nNo lines: %d\n", noTrainVP, noLines);
        exit(1);
    }
    vpFeatSize = featDim;

    // Closing the file
    fclose(filePt);
}

// Function to tokenize the training sentences and link to word2vec vocab
void tokenizeTrainSentences(){
    long i;
    for (i = 0; i < noTrainVP; i++){
        // Copy the word into a local variable line
        char* line = (char*) malloc(MAX_SENTENCE);
        strcpy(line, train[i].sent);

        int count = 0, n;

        // Split based on 's
        char* first = multi_tok(line, "'s");
        char* second = multi_tok(NULL, "'s");

        // Join both the parts without the 's (from baseline: add it at the end)
        if(second != NULL) line = strcat(first, strcat(second, " \'s"));
        else line = first;

        char* temp = (char*) malloc(MAX_SENTENCE);
        strcpy(temp, line);
        
        // Remove ' ', ',', '.', '?', '!', '\', '/'
        char* delim = " .,/!?\\";
        line = strtok(line, delim);
        // Going over the line to determine the number of parts
        while(line != NULL){
            count++;
            line = strtok(NULL, delim);
        }

        // Now store the word components, looping over them
        train[i].count = count;
        train[i].index = (long*) malloc(train[i].count * sizeof(long));

        line = strtok(temp, delim);
        count = 0;
        while(line != NULL){
            // Convert the token into lower case
            for(n = 0; line[n]; n++) line[n] = tolower(line[n]);
           
            // Save the index
            train[i].index[count] = SearchVocab(line);
            line = strtok(NULL, delim);
            count++;
        }
    }
}
