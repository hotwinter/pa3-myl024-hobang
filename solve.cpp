/* 
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 * 
 * Modified and  restructured by Scott B. Baden, UCSD
 * 
 */

#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include "time.h"
#include "apf.h"
#include "Plotting.h"
#include "cblock.h"
#include <emmintrin.h>
#include <mpi.h>
using namespace std;

extern int my_rank;
extern int my_pi, my_pj;
extern int my_m, my_n;

extern double *in_W, *in_E, *out_W, *out_E;

extern control_block cb;

void repNorms(double l2norm, double mx, double dt, int m,int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
void printMat(const char*, double*, int m, int n);

#define CORNER_SIZE 1
#define PAD_SIZE 2
#define ROOT 0

#define FARTHEST_NORTH 0
#define FARTHEST_WEST 0

enum { NORTH = 0, EAST, WEST, SOUTH };

#ifdef SSE_VEC
// If you intend to vectorize using SSE instructions, you must
// disable the compiler's auto-vectorizer
__attribute__((optimize("no-tree-vectorize")))
#endif 

// The L2 norm of an array is computed by taking sum of the squares
// of each element, normalizing by dividing by the number of points
// and then taking the sequare root of the result
//
double L2Norm(double sumSq)
{
	double l2norm = sumSq /  (double) ((cb.m)*(cb.n));
	l2norm = sqrt(l2norm);
	
	return l2norm;
}

void communicate(double *E_prev)
{
	const int FARTHEST_SOUTH = cb.py - 1;
	const int FARTHEST_EAST = cb.px - 1; 
	
	int i, j;
	MPI_Request sendReqs[4];
	MPI_Request recvReqs[4];
	MPI_Status  statuses[4];
	
	int msgCounter = 0;

	if (my_pi == FARTHEST_NORTH)
	{
		for (i = (0 + CORNER_SIZE); i < my_n + CORNER_SIZE; ++i)
		{
			E_prev[i] = E_prev[i + 2*(my_n + PAD_SIZE)];
		}
	}
	else 	//Send the NORTH boundary & fill the NORTH ghost cells
	{
		MPI_Irecv(&E_prev[CORNER_SIZE], my_n, MPI_DOUBLE, my_rank - cb.px, SOUTH, MPI_COMM_WORLD, recvReqs + msgCounter);
		MPI_Isend(&E_prev[my_n + PAD_SIZE+CORNER_SIZE], my_n, MPI_DOUBLE, my_rank - cb.px, NORTH, MPI_COMM_WORLD, sendReqs + 0);

		msgCounter++;
	}

	if (my_pi == FARTHEST_SOUTH)
	{
		for (i = (my_m + CORNER_SIZE)*(my_n + PAD_SIZE) + CORNER_SIZE; i < (my_m + PAD_SIZE)*(my_n + PAD_SIZE) - CORNER_SIZE; ++i)
		{
			E_prev[i] = E_prev[i - 2*(my_n + PAD_SIZE)];
		}
	}
	else	// Send the SOUTH boundary & fill the SOUTH ghost cells
	{
		MPI_Irecv(&E_prev[(my_m + CORNER_SIZE)*(my_n + PAD_SIZE) + CORNER_SIZE], my_n, MPI_DOUBLE, my_rank + cb.px, NORTH, MPI_COMM_WORLD, recvReqs + msgCounter);
		MPI_Isend(&E_prev[my_m*(my_n + PAD_SIZE) + CORNER_SIZE], my_n, MPI_DOUBLE, my_rank + cb.px, SOUTH, MPI_COMM_WORLD, sendReqs + 1);

		msgCounter++;
	}

	if (my_pj == FARTHEST_WEST) 
	{
		for (i = my_n + PAD_SIZE; i < (my_m + CORNER_SIZE)*(my_n + PAD_SIZE); i += (my_n + PAD_SIZE))
		{
			E_prev[i] = E_prev[i + 2];
		}
	}
	else	// Send the WEST boundary & fill the WEST ghost cells
	{
		for (i = my_n + PAD_SIZE + 1, j = 0; j < my_m; i += my_n + PAD_SIZE, ++j)
		{
			out_W[j] = E_prev[i];
		}

		MPI_Irecv(in_W, my_m, MPI_DOUBLE, my_rank - 1, EAST, MPI_COMM_WORLD, recvReqs + msgCounter);
		MPI_Isend(out_W, my_m, MPI_DOUBLE, my_rank - 1, WEST, MPI_COMM_WORLD, sendReqs + 2);
	
		msgCounter++;
	}

	if (my_pj == FARTHEST_EAST)
	{
		for (i = (my_n + CORNER_SIZE) + 1*(my_n + PAD_SIZE); i < (my_n + CORNER_SIZE) + (my_m + CORNER_SIZE)*(my_n + PAD_SIZE); i += (my_n + PAD_SIZE))
		{
			E_prev[i] = E_prev[i - 2];
		}
	}
	else	// Send the EAST boundary & fill the EAST ghost cells
	{
		for (i = my_n + (my_n + PAD_SIZE), j = 0; j < my_m; i += (my_n + PAD_SIZE), ++j)
		{
			out_E[j] = E_prev[i];
		}

		MPI_Irecv(in_E, my_m, MPI_DOUBLE, my_rank + 1, WEST, MPI_COMM_WORLD, recvReqs + msgCounter);
		MPI_Isend(out_E, my_m, MPI_DOUBLE, my_rank + 1, EAST, MPI_COMM_WORLD, sendReqs + 3);

		msgCounter++;
	}

	// Wait for all messages to be received before unpacking data for WEST and EAST messages
	MPI_Waitall(msgCounter, recvReqs, statuses);

	if (my_pj != FARTHEST_WEST)
	{
		for (i = my_n + PAD_SIZE, j = 0; j < my_m; i += my_n + PAD_SIZE, ++j) 		
		{
			E_prev[i] = in_W[j];
		}
	}

	if (my_pj != FARTHEST_EAST)
	{
		for (i = (my_n + CORNER_SIZE) + (my_n + PAD_SIZE), j = 0; j < my_m; i += my_n + PAD_SIZE, ++j)
		{
			E_prev[i] = in_E[j];
		}
	}

	//printf("RANK %d's E_prev with ghost cells filled:\n", my_rank);
	//printMat("", E_prev, my_m, my_n);
}


void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf)
{
	// Simulated time is different from the integer timestep number
	double t = 0.0;
	
	double *E = *_E, *E_prev = *_E_prev;
	double *R_tmp = R;
	double *E_tmp = *_E;
	double *E_prev_tmp = *_E_prev;
	double mx, sumSq;
	int niter;
	//int m = cb.m, n=cb.n;
	int innerBlockRowStartIndex = (my_n+PAD_SIZE)+CORNER_SIZE;
	int innerBlockRowEndIndex = (((my_m+PAD_SIZE)*(my_n+PAD_SIZE) - CORNER_SIZE) - (my_n)) - (my_n+PAD_SIZE);

	// We continue to sweep over the mesh until the simulation has reached
	// the desired number of iterations
	for (niter = 0; niter < cb.niters; niter++)
	{
		if (cb.plot_freq)
		{
			//plotter->updatePlot(E,  -1, m+1, n+1);
		}

		if(!cb.noComm)
		{
			communicate(E_prev);
		} 
		//////////////////////////////////////////////////////////////////////////////

		// #define FUSED 1

	#ifdef FUSED
		// Solve for the excitation, a PDE
		for (int j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+= my_n + PAD_SIZE)
		{
			E_tmp = E + j;
			E_prev_tmp = E_prev + j;
			R_tmp = R + j;

			for (int i = 0; i < my_n; i++)
			{
				E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(my_n+2)]+E_prev_tmp[i-(my_n+2)]);
				E_tmp[i] += -dt*(kk*E_tmp[i]*(E_tmp[i]-a)*(E_tmp[i]-1)+E_tmp[i]*R_tmp[i]);
				R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_tmp[i]+M2))*(-R_tmp[i]-kk*E_tmp[i]*(E_tmp[i]-b-1));
			}
		}
	#else
		// Solve for the excitation, a PDE
		for (int j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += my_n + PAD_SIZE)
		{
			E_tmp = E + j;
			E_prev_tmp = E_prev + j;

			for (int i = 0; i < my_n; ++i)
			{
				E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(my_n+2)]+E_prev_tmp[i-(my_n+2)]);
			}
		}

		/* 
		 * Solve the ODE, advancing excitation and recovery variables
		 *     to the next timtestep
		 */

		for (int j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += my_n + PAD_SIZE)
		{
			E_tmp = E + j;
			R_tmp = R + j;
			for (int i = 0; i < my_n; ++i)
			{
				E_tmp[i] += -dt*(kk*E_tmp[i]*(E_tmp[i]-a)*(E_tmp[i]-1)+E_tmp[i]*R_tmp[i]);
				R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_tmp[i]+M2))*(-R_tmp[i]-kk*E_tmp[i]*(E_tmp[i]-b-1));
			}
		}
	#endif
		/////////////////////////////////////////////////////////////////////////////////

		if (cb.plot_freq)
		{
			if (!(niter % cb.plot_freq))
			{
				//plotter->updatePlot(E,  niter, m, n);
			}
		}

		// Swap current and previous meshes
		double *tmp = E; 
		E = E_prev; 
		E_prev = tmp;
	} //end of 'niter' loop at the beginning
	
	double reducedSq = 0.0;	

	stats(E_prev, my_m, my_n, &mx, &sumSq);	

	MPI_Reduce(&sumSq, &reducedSq, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);
	MPI_Reduce(&mx, &Linf, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);

	L2 = L2Norm(reducedSq);

	// Swap pointers so we can re-use the arrays
	*_E = E;
	*_E_prev = E_prev;
	
}
