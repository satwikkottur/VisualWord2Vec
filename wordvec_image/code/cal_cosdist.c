//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <vector>
#include <cstring>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

int main(int argc, char **argv) {
  FILE *f;
  char st1[max_size];
  char *bestw[N];
  char file_name[max_size], st[100][max_size];
  float dist, len, bestd[N], vec[max_size], vec2[max_size];
  long long words, size, i, a, b, c, d, cn, bi[100], num_pair, id1, id2;
  char ch;
  float *M;
  char *vocab;

  if (argc < 3) {
    printf("Usage: ./distance <FILE>\nwhere FILE contains word projections in the BINARY FORMAT\n");
    return 0;
  }
	
	//read vector
  strcpy(file_name, argv[1]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  printf("words/size is %lld/%lld\n",words,size);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
  for (a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
  M = (float *)malloc((long long)words * (long long)size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  }
  for (b = 0; b < words; b++) {
    fscanf(f, "%s%c", &vocab[b * max_w], &ch);
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
  }
  fclose(f);

	// read words
  string dir_words = argv[2];
	char *input = (char*)dir_words.c_str();

	ifstream infile(input);
	vector <string> record;
	while(infile) {
		string s;
		if(!getline(infile,s))
			break;
		istringstream ss(s);
		while(ss) {
			string s;
			if(!getline(ss,s,'\t'))
				break;
			//num = atof(s.c_str());
			record.push_back(s.c_str());
			//printf("%s,",s.c_str());			
		}
		//printf("\n");
	}	
	if(!infile.eof())
		printf("wrong file\n");
	printf("size of words pair %d\n",record.size());

  /*
  string dir_model_out = argv[3]; 
  char *output = (char*)dir_model_out.c_str();   
  FILE *f3=fopen(output, "w");
  
  for (a = 0; a < record.size(); a++) {
    string str = record.at(a);
    const char *pStr = str.c_str();
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], pStr)) break;
    if (b == words) b = -1;
    bi[a] = b;
    printf("\nWord: %s  Position in vocabulary: %lld\n", str.c_str(), bi[a]);
    if (b == -1) {
      printf("Out of dictionary word!\n");
      fprintf(f3,"0\n");    
      continue;
    }
    else{
      for(i = 0; i<size;i++) vec[i] = 0;
      for(i = 0; i<size;i++) vec[i] += M[i + bi[a] * size];
      for(i = 0; i<size;i++) fprintf(f3,"%f,",vec[i]);
    }
    fprintf(f3,"\n");       
  }
  fclose(f3);
  */

  string dir_model_out = argv[3]; 
  char *output = (char*)dir_model_out.c_str();   
  FILE *f3=fopen(output, "w");
  num_pair = record.size()/3;
  printf("number of comparison pair %lld\n", num_pair);
  for (i = 0; i < num_pair; i++) {
    string str1 = record.at(i * 3);
    string str2 = record.at(i * 3 + 1);
    string str3 = record.at(i * 3 + 2);
    dist = 0;
    //printf("%s %s %s \n", str1.c_str(), str2.c_str(), str3.c_str());
    const char *pStr = str1.c_str();
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], pStr)) break;
    if (b == words) b = -1;
    id1 = b;
    //printf("\nWord: %s  Position in vocabulary: %lld\n", str1.c_str(), id1);
    if (b == -1) {
      printf("%s: Out of dictionary word!\n", str1.c_str());
      //fprintf(f3,"0\n");    
      dist = 2;
      continue;
    } else{
      for(a = 0; a < size; a++) vec[a] = 0;
      for(a = 0; a < size; a++) vec[a] += M[a + id1 * size];
    }

    const char *pStr2 = str2.c_str();
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], pStr2)) break;
    if (b == words) b = -1;
    id2 = b;
    //printf("\nWord: %s  Position in vocabulary: %lld\n", str2.c_str(), id2);
    if (b == -1) {
      printf(" %s: out of dictionary word!\n", str2.c_str());
      //fprintf(f3,"0\n");    
      dist = 2;
      continue;
    } else{
      for(a = 0; a < size; a++) vec2[a] = 0;
      for(a = 0; a < size; a++) vec2[a] += M[a + id2 * size];
    }
    for (a = 0; a < size; a++) {
      dist += vec[a] * vec2[a];
    } 
    printf("%s, %s, %f\n", str1.c_str(), str2.c_str(), dist);
    fprintf(f3,"%s, %f\n",str3.c_str(), dist);
  }
  fclose(f3);

  return 0;
}