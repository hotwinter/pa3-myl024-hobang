/* 
 * Utilities for the Aliev-Panfilov code
 *
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 * 
 * Modified and  restructured by Scott B. Baden, UCSD, 11/2/2015
 */

#include <cstdlib>
#include <stdlib.h>
#include <malloc.h>
#include <iostream>
#include <assert.h>
#include <iomanip>
#include <string>
#include <math.h>
#include "apf.h"

using namespace std;

void printMat(const char mesg[], double *E, int m, int n);

//
// Initialization
//
// We set the right half-plane of E_prev to 1.0, the left half plane to 0
// We set the botthom half-plane of R to 1.0, the top half plane to 0
// These coordinates are in world (global) coordinate and must
// be mapped to appropriate local indices when parallelizing the code
//
void init (double *E,double *E_prev,double *R,int m,int n){
    int i;

    for (i=0; i < (m+2)*(n+2); i++)
        E_prev[i] = R[i] = 0;

    for (i = (n+2); i < (m+1)*(n+2); i++) {
	int colIndex = i % (n+2);		// gives the base index (first row's) of the current index

        // Need to compute (n+1)/2 rather than n/2 to work with odd numbers
	if(colIndex == 0 || colIndex == (n+1) || colIndex < ((n+1)/2+1))
	    continue;

        E_prev[i] = 1.0;
    }

    for (i = 0; i < (m+2)*(n+2); i++) {
	int rowIndex = i / (n+2);		// gives the current row number in 2D array representation
	int colIndex = i % (n+2);		// gives the base index (first row's) of the current index

        // Need to compute (m+1)/2 rather than m/2 to work with odd numbers
	if(colIndex == 0 || colIndex == (n+1) || rowIndex < ((m+1)/2+1))
	    continue;

        R[i] = 1.0;
    }
    // We only print the meshes if they are small enough
    printMat("E_prev",E_prev,m,n);
    printMat("R",R,m,n);
}

//
// Report statistics about the computation: the sums of the squares
// and the max value of the mesh
// These values should not vary (except to within roundoff)
// when we use different numbers of processes to solve the problem

// We use the sum of squares to compute the L2 norm, which is a normalized
// square root of this sum; see L2Norm() in Helper.cpp

void stats(double *E, int m, int n, double *_mx, double *sumSq){
     double mx = -1;
     double _sumSq = 0;
     int i, j;

     for (i=0; i< (m+2)*(n+2); i++) {
        int rowIndex = i / (n+2);			// gives the current row number in 2D array representation
        int colIndex = i % (n+2);		// gives the base index (first row's) of the current index		

        if(colIndex == 0 || colIndex == (n+1) || rowIndex == 0 || rowIndex == (m+1))
            continue;

        _sumSq += E[i]*E[i];
        double fe = fabs(E[i]);
        if (fe > mx)
            mx = fe;
    }
    *_mx = mx;
    *sumSq = _sumSq;
}
