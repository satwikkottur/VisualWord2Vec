#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char* argv[]){
    FILE *filePt;
    filePt = fopen("/home/satwik/VisualWord2Vec/data/PSR_features.txt", "rb");

    if(filePt == NULL){
        printf("Can't find file!\n");
        exit(1);
    }

    char p[100], s[100], r[100];
    while(fscanf(filePt, "<%[^<>:]:%[^<>:]:%[^<>:]>\n", p, s, r) != EOF)
        printf("%s  %s  %s\n", p, s, r);

    return 0;
}
