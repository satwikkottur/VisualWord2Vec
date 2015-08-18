#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int main(int argc, char* argv[]){
    //char* word = "girl's hand";
    char* word;
    word = (char*) malloc(50);
    strcpy(word, "Pizza / Cheese");
    //strcpy(word, "girl's hand");

    // Split based on 's
    char* first = multi_tok(word, "'s");
    char* second = multi_tok(NULL, "'s");

    // Join both the parts without the 's
    if(second != NULL) word = strcat(first, second);
    
    // Remove ' ', ',', '.', '?', '!', '\', '/'
    first = strtok(word, " .,/!?\\");
    printf("%s\n", first);

    /*char* first = multi_tok(word, delim);
    char* second = multi_tok(NULL, delim);
    printf("%s %s\n", first, second);
    while(token != NULL){
        printf("%s\n", token);
        token = multi_tok(NULL, delim);
        token = strtok(word, ",.?!\\/ ");
    }

    token = strtok(word, ",.?!\\/ ");
    printf("%s\n", token);*/
    return 0;
}
