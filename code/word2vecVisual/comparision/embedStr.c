// Function to spit out effect on all the feature words
// and see the difference
//
#define MAX_STRING 100
# include <string.h>
# include <stdlib.h>
# include <stdio.h>
# include <ctype.h>

void splitFeatureWords(FILE*, char*);
char* multi_tok(char*, char*);

int main(){
    // Read the feature word vocab
    FILE* featPt = fopen("vocab.txt", "rb");
    if(featPt == NULL) printf("No file\n");
    
    // Open a file to write the splits of feature word
    FILE* splitPt = fopen("c_splitting.txt", "wb");

    char* featWord = (char*) malloc(MAX_STRING);
    while(fscanf(featPt, "%[^\n]\n", featWord) != EOF){
        // Write the splits
        splitFeatureWords(splitPt, featWord);
    }
    
    // close the file
    fclose(splitPt);
    fclose(featPt);
    //splitFeatureWords(NULL, "children's hands");
    
    return 1;
}

void splitFeatureWords(FILE* filePt, char* word){
    //printf("%s = ", word);
    fprintf(filePt, "%s = ", word);
    int count=0, i;

    // Split based on 's
    char* token = (char*) malloc(MAX_STRING);
    strcpy(token, word);

    char* first = multi_tok(token, (char*)"'s");
    char* second = multi_tok(NULL, (char*)"'s");

    // Join both the parts without the 's
    if(second != NULL)
        token = strcat(first, strcat(second, " \'s"));

    else token = first;

    char* temp = (char*) malloc(MAX_STRING);
    strcpy(temp, token);
    
    // Remove ' ', ',', '.', '?', '!', '\', '/'
    char* delim = (char*)" .,/!?\\";
    token = strtok(token, delim);
    // Going over the token to determine the number of parts
    while(token != NULL){
        count++;
        token = strtok(NULL, delim);
    }

    token = strtok(temp, delim);
    while(token != NULL){
        // Convert the token into lower case
        for(i = 0; token[i]; i++) token[i] = tolower(token[i]);
       
        // Save to the file
        fprintf(filePt, ":%s:", token);
        token = strtok(NULL, delim);
    }

    //printf("\n");
    fprintf(filePt, "\n");
}

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
