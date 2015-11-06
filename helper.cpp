/* 
 * Utilities for the Aliev-Panfilov code
 * Scott B. Baden, UCSD
 * Nov 2, 2015
 */

#include <iostream>
#include <assert.h>
// Needed for memalign
#include <malloc.h>

using namespace std;

double *alloc1D(int m,int n){
    int nx=n, ny=m;
    double *E;
    // Ensures that allocatdd memory is aligned on a 16 byte boundary
    assert(E= (double*) memalign(16, sizeof(double)*nx*ny) );
    return(E);
}

void printMat(const char mesg[], double *E, int m, int n){
    int i;
    if (m>8)
      return;
    printf("%s\n",mesg);
    for (i=0; i < (m+2)*(n+2); i++){
       int rowIndex = i / (n+2);
       int colIndex = i % (n+2);
       if ((colIndex>0) && (colIndex<m+1))
          if ((rowIndex > 0) && (rowIndex < n+1))
            printf("%6.3f ", E[i]);
       if (colIndex == m+1)
	    printf("\n");
    }
}
