# include "helperFunctions.h"

// Saving the word2vec vectors for further use
void saveWord2Vec(char* fileName){
    FILE* filePt = fopen(fileName, "wb");

    long i, j, offset;
    for (i = 0; i < vocab_size; i++){
        offset = i * layer1_size;
        fprintf(filePt, "%s ", vocab[i].word);
        for (j = 0; j < layer1_size - 1; j++)
            fprintf(filePt, "%f ", syn0[offset + j]);

        fprintf(filePt, "%f\n", syn0[offset + layer1_size - 1]);
    }

    fclose(filePt);
}

// Load the word2vec vectors
// We assume all the other parameters(vocabulary is kept constant)
// Use with caution
void loadWord2Vec(char* fileName){
    FILE* filePt = fopen(fileName, "rb");
    
    long i, j, offset;
    long noVocab, dims;
    float value;
    char word[MAX_STRING];
    fscanf(filePt, "%ld %ld\n", &noVocab, &dims);
    if(vocab_size != noVocab && layer1_size != dims){
        printf("Word2Vec reading incompatible! \n");
        exit(1);
    }

    // Reading the dimensions
    for (i = 0; i < vocab_size; i++){
        fscanf(filePt, "%s", word);
        // Allocate memory and store the word
        offset = layer1_size * i;
        for(j = 0; j < layer1_size; j++){
            fscanf(filePt, "%f", &value);

            // Storing the value
            syn0[offset + j] = value;
        }
        //printf("%s\n", word);
    }

    fclose(filePt);
}

// Multiple character split
// Source: http://stackoverflow.com/questions/29788983/split-char-string-with-multi-character-delimiter-in-c
char *multi_tok(char *input, char *delimiter) {
    static char *string;
    if (input != NULL)
        string = input;

    if (string == NULL)
        return string;

    char *end = strstr(string, delimiter);
    if (end == NULL) {
        char *temp = string;
        string = NULL;
        return temp;
    }

    char *temp = string;

    *end = '\0';
    string = end + strlen(delimiter);
    return temp;
}

