// Definitions for functions related to the visual paraphrasing task
# include "vpFunctions.h"

// Globals for the current scope
long noTrainVP = 0;
int vpFeatSize = 0;
struct trainSent* train;

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
    while(fscanf(filePtr, "%[^\n]\n", currentSent) != EOF)
        sentCount++;
   
   // Rewind the stream and read again
   rewind(filePtr);
    
    // Allocate the train
    train = (struct trainSent*) malloc(sizeof(struct trainSent) * sentCount);
    // Read and store contents
    for( i = 0; i < sentCount; i++){
        if(fscanf(filePtr, "%[^\n]\n", currentSent) != EOF){
            // Allocate memory and copy sentence
            train[i].sent = (char*) malloc(sizeof(char) * MAX_SENTENCE);
            strcpy(train[i].sent, currentSent);
            
            // TODO:
            // Cross refer to the word2vec vocab later
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
    printf("File read with %ld sentences!....\n\n", sentCount);
}

void readVPVisualFeatures(char* visualPath){
    // 

}
