/* randompermute.c - A program will generate "random permutations of n elements"
   if at all points the n! possible permutations have all the same probability 
   of being generated.
*/
// Source: 
// http://cis-linux1.temple.edu/~giorgio/cis71/code/randompermute.c

#include <stdio.h>
#include <stdlib.h>

// It returns a random permutation of 0..n-1
void rpermute(int n, int* a) {
    int k;
    for (k = 0; k < n; k++)
	a[k] = k;
    for (k = n-1; k > 0; k--) {
        int j = rand() % (k+1);
        int temp = a[j];
        a[j] = a[k];
        a[k] = temp;
    }
}
/*void printarray(int n, int* a) {
    int k = 0;
    for (k = 0; k < n; k++) {
	printf("%6d   ", a[k]);
	if (k % 8 == 7)
	    printf("\n");
    } 
}

int main(void) {
    int limit = 10000;
    int *a;
    int k;
    // Print 7 permutations
    for (k = 0; k < 1; k++) {
	a = rpermute(limit);
	//printarray(limit, a);
	printf("\n");
    }
    
    return 0;
}*/
