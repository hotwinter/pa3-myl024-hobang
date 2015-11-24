/* 
 * Utilities for the Aliev-Panfilov code
 * Scott B. Baden, UCSD
 * Nov 2, 2015
 */

#include <iostream>
#include <assert.h>
#include <malloc.h>
#include <math.h>
#include "cblock.h"
#include <mpi.h>
#include <string.h>
using namespace std;

extern control_block cb;

int my_rank;
int my_pi, my_pj; // position of the process in the x-by-y grid of processes (we distribute the processes in row major order)
int my_m, my_n; // the number of rows,cols in the subproblem (respectively)
int my_stride;

// We allocate some space for WESTward and EASTward messages
// Because the ghosts cells are noncontiguous, we need to pack
// and unpack the data when sending and receiving. I believe
// that we can also use MPI_DOUBLE_vector to specify a "strided"
// data type for the messages, but http://stackoverflow.com/a/29134807
// suggests that it's just better to do the (un)packing ourselves.
double *in_W, *in_E, *out_W, *out_E;

void printMat(const char mesg[], double *E);

inline int min(int a, int b) { if (a < b) return a; else return b; }
inline int max(int a, int b) { if (a > b) return a; else return b; }

//
// Initialization
//
// We set the right half-plane of E_prev to 1.0, the left half plane to 0
// We set the botthom half-plane of R to 1.0, the top half plane to 0
// These coordinates are in world (global) coordinate and must
// be mapped to appropriate local indices when parallelizing the code
//
void init (double *E, double *E_prev, double *R, int m, int n)
{
	// By now, global variables my_rank, my_m, my_n have already been set

	int nMin = cb.n / cb.px;
	int mMin = cb.m / cb.py;

	// Number of block-rows/cols that are extended by one
	int rx = cb.n % cb.px;
	int ry = cb.m % cb.py;

	///////////////////////////////////////////////////////////////////////////////////////
	//// Initialize R (R's ghosts cells are arbitrary since we don't use them anyways) ////
	///////////////////////////////////////////////////////////////////////////////////////

	int iMin = my_pi*mMin + min(my_pi, ry); // GLOBAL index of the first row of the "computational" block (ignoring ghost cells)
	int di = min((cb.m + 1)/2 - iMin, my_m); // Distance from iMin to the first row of 1.0s

	for (int i = 0; i < (my_m + 2)*my_stride; ++i)
	{
		int rowIndex = i / my_stride;
		int colIndex = i % my_stride;

		if (rowIndex == 0 || rowIndex >= my_m + 1 || rowIndex <= di ||
			colIndex == 0 || colIndex >= my_n + 1)
		{
			R[i] = 0.0;
		}
		else
		{
			R[i] = 1.0;
		}
	}

	///////////////////////////////////////////////////////////////////////////////
	//// Initialize E_prev (E_prev's ghost cells are initialized to 0)         ////
	//// Msg passing will fill in interior ghost cells with appropriate values ////
	///////////////////////////////////////////////////////////////////////////////

	int jMin = my_pj*nMin + min(my_pj, rx);
	int dj = min((cb.n + 1)/2 - jMin, my_n);

	for (int i = 0; i < (my_m + 2)*my_stride; ++i)
	{
		int rowIndex = i / my_stride;
		int colIndex = i % my_stride;

		if (colIndex == 0 || colIndex >= my_n + 1 || colIndex <= dj ||
			rowIndex == 0 || rowIndex >= my_m + 1)
		{
			E_prev[i] = 0.0;
		}
		else
		{
			E_prev[i] = 1.0;
		}
	}

	// We only print the meshes if they are small enough
	
//	printf("\n\nRANK %d INITIAL CONDITIONS:\n\n", my_rank);
//	printf("RANK %d my_stride is: %d\n", my_rank, my_stride);
//	printMat("E_prev", E_prev);
//	printMat("R", R);
}

// NOTE: This gets called with arguments (cb.m+2, cb.n+2) in apf.cpp
double *alloc1D(int mPlus2,int nPlus2)
{
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	int m = mPlus2 - 2; // get the unpadded size of the global problem
	int n = nPlus2 - 2;

	my_pi = my_rank / cb.px;
	my_pj = my_rank % cb.px;

	my_m = m / cb.py + (my_pi < (m % cb.py));
	my_n = n / cb.px + (my_pj < (n % cb.px));

	my_stride = (my_n % 2 == 0) ? my_n + 2 : my_n + 3;

	// Allocate contiguous memory for the WESTward and EASTward messages
	in_W  = new double[4*my_m];
	in_E  = in_W  + my_m;
	out_W = in_E  + my_m;
	out_E = out_W + my_m;

	// Ensure that allocated memory is aligned on a 16 byte boundary
	// Pad the subproblem to accomodate E_prev's ghost cells
	// We also pad R, although this is redundant and just for consistency
	double *E;
	assert(E = (double*)memalign(16, sizeof(double)*((my_m + 2)*my_stride + 1)));
	E++;

	return(E);
}

void printMat(const char mesg[], double *E)
{
	if (my_m > 8)
	{
		//return;
	}
	printf("%s\n", mesg);

	for (int i = 0; i < (my_m + 2)*my_stride; ++i)
	{
		int rowIndex = i / my_stride;
		int colIndex = i % my_stride;

		if ((colIndex == 0 || colIndex >= my_n + 1) && (rowIndex == 0 || rowIndex == my_m + 1))
		{
			printf("      ");
		}
		else
		{
			printf("%1.3f ", E[i]); // For testing purposes, we also print the ghost cells
		}

		if (colIndex == my_stride - 1)
		{
			printf("\n");
		}
	}
}

// We overload this to facillitate our padded vectorization
void stats(double *E, double *_mx, double *sumSq)
{
	double mx = -1;
	double _sumSq = 0;
 
	for (int i = 1 + my_stride; i < (my_m + 1)*my_stride; ++i)
	{
		int rowIndex = i / my_stride;           // gives the current row number in 2D array representation
		int colIndex = i % my_stride;       // gives the base index (first row's) of the current index      
 
		if (colIndex == 0 || colIndex >= my_n +1)
		{
			continue;
		}
 
		_sumSq += E[i]*E[i];
		double fe = fabs(E[i]);

		if (fe > mx)
		{
			mx = fe;
		}
	}
	*_mx = mx;
	*sumSq = _sumSq;
}
