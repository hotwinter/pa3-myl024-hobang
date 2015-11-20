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

extern control_block cb;

void repNorms(double l2norm, double mx, double dt, int m,int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);

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

// The following enumerates the MPI tags that we use to
// distinguish between the boundaries from which a message
// is SENT. Together with the SOURCE process rank, we can
// determine exactly what data is held in an MPI messge.
// e.g. {SOURCE RANK = 0, tag = EAST} indicates that the
//      message holds the EAST computational boundary, to
//      be used for filling the WEST ghost cells of process 1.
enum { NORTH = 0, EAST, WEST, SOUTH };

// Fills in the padded regions of the process:
// - If one of the four process boundaries is also a GLOBAL boundary
//   then we enforce that the gradient towards the boundary is zero.
//   In this case, we don't send or recieve messages.
// - For INTERIOR process boundaries, we send the boundary computational
//   cells to the neighboring process, which also produces the data for
//   the boundary ghost cells. 

void send(double *E_prev)
{

}

void recv(double *E_prev)
{

}

void communicate(double *E_prev)
{
	int i, j;
	MPI_Request sendReqs[4];
	MPI_Request recvReqs[4];

	double* in_W = new double[2*my_m]; // allocate some space for the received messages
	double* in_E = in_W + my_m;

	double* out_W = new double[2*my_m]; // allocate some space for the sent messages
	double* out_E = out_W + my_m;

	int msgCounter = 0;

	// Send the NORTH boundary & fill the NORTH ghost cells
	if (my_pi == 0)
	{
		for (i = 1; i < my_n + 1; ++i)
		{
			//printf("N  %d  %d\n", i, i+2*(my_n + 2));
			E_prev[i] = E_prev[i + 2*(my_n + 2)];
		}
	}
	else
	{
		MPI_Irecv(&E_prev[1], my_n, MPI_DOUBLE, my_rank - cb.px, SOUTH, MPI_COMM_WORLD, recvReqs + msgCounter);
		MPI_Isend(&E_prev[my_n + 3], my_n, MPI_DOUBLE, my_rank - cb.px, NORTH, MPI_COMM_WORLD, sendReqs + 0);
		msgCounter++;
	}

	// Send the SOUTH boundary & fill the SOUTH ghost cells
	if (my_pi == cb.py - 1)
	{
		for (i = (my_m + 1)*(my_n + 2) + 1; i < (my_m + 2)*(my_n + 2) - 1; ++i)
		{
			E_prev[i] = E_prev[i - 2*(my_n + 2)];
		}
	}
	else
	{
		//printf("S\n");
		MPI_Irecv(&E_prev[(my_m + 1)*(my_n + 2) + 1], my_n, MPI_DOUBLE, my_rank + cb.px, NORTH, MPI_COMM_WORLD, recvReqs + msgCounter);
		MPI_Isend(&E_prev[my_m*(my_n + 2) + 1], my_n, MPI_DOUBLE, my_rank + cb.px, SOUTH, MPI_COMM_WORLD, sendReqs + 1);
		msgCounter++;
		//printf("SS\n");
	}

	// Send the WEST boundary & fill the WEST ghost cells
	if (my_pj == 0)
	{
		for (i = my_n + 2; i < (my_m + 1)*(my_n + 2); i += my_n + 2)
		{
			E_prev[i] = E_prev[i + 2];
		}
	}
	else
	{
		for (i = my_n + 3, j = 0; j < my_m; i += my_n + 2, ++j)
		{
			//printf("W  %d  %d\n", i, j);
			out_W[j] = E_prev[i];
		}
		MPI_Irecv(in_W, my_m, MPI_DOUBLE, my_rank - 1, EAST, MPI_COMM_WORLD, recvReqs + msgCounter);
		MPI_Isend(&E_prev[my_n + 3], my_m, MPI_DOUBLE, my_rank - 1, WEST, MPI_COMM_WORLD, sendReqs + 2);
		msgCounter++;
	}

	// Send the EAST boundary & fill the EAST ghost cells
	if (my_pj == cb.px - 1)
	{
		for (i = (my_n + 1) + 1*(my_n + 2); i < (my_n + 1) + (my_m + 1)*(my_n + 2); i += my_n + 2)
		{
			//printf("E  %d  %d  %d\n", my_m, my_n, i);
			E_prev[i] = E_prev[i - 2];
		}
	}
	else
	{
		for (i = (my_n + 1) + (my_n + 2), j = 0; j < my_m; i += my_n + 2, ++j)
		{
			out_E[j] = E_prev[i];
		}
		MPI_Irecv(in_E, my_m, MPI_DOUBLE, my_rank + 1, WEST, MPI_COMM_WORLD, recvReqs + msgCounter);
		MPI_Isend(out_E, my_m, MPI_DOUBLE, my_rank + 1, EAST, MPI_COMM_WORLD, sendReqs + 3);
		msgCounter++;
	}

	//printf("lala\n");

	MPI_Status statuses[4];
	MPI_Waitall(msgCounter, recvReqs, statuses);

	//printf("asdfasdf\n");

	if (my_pj != 0)
	{
		for (i = my_n + 2, j = 0; j < my_m; i += my_n + 2, ++j) 		
		{
			//printf("blahblah\n");
			E_prev[i] = in_W[j];
		}
	}

	if (my_pj != cb.px - 1)
	{
		for (i = (my_n + 1) + (my_n + 2), j = 0; j < my_m; i += my_n + 2, ++j)
		{
			//printf("haha\n");
			E_prev[i] = in_E[j];
		}
	}

	delete[] in_W;
	delete[] out_W;

	printf("communication finished\n");
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
	int m = cb.m, n=cb.n;
	int innerBlockRowStartIndex = (my_n+2)+1;
	int innerBlockRowEndIndex = (((my_m+2)*(my_n+2) - 1) - (my_n)) - (my_n+2);


	// We continue to sweep over the mesh until the simulation has reached
	// the desired number of iterations
	for (niter = 0; niter < cb.niters; niter++)
	{
		if (cb.debug && (niter==0))
		{
			stats(E_prev,m,n,&mx,&sumSq);
			double l2norm = L2Norm(sumSq);
			repNorms(l2norm,mx,dt,m,n,-1, cb.stats_freq);

			if (cb.plot_freq)
			{
				plotter->updatePlot(E,  -1, m+1, n+1);
			}
		}

		int i, j;

		communicate(E_prev);

		//////////////////////////////////////////////////////////////////////////////

		// #define FUSED 1

	#ifdef FUSED
		// Solve for the excitation, a PDE
		for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+= my_n + 2)
		{
			E_tmp = E + j;
			E_prev_tmp = E_prev + j;
			R_tmp = R + j;

			for (i = 0; i < my_n; i++)
			{
				E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(my_n+2)]+E_prev_tmp[i-(my_n+2)]);
				E_tmp[i] += -dt*(kk*E_tmp[i]*(E_tmp[i]-a)*(E_tmp[i]-1)+E_tmp[i]*R_tmp[i]);
				R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_tmp[i]+M2))*(-R_tmp[i]-kk*E_tmp[i]*(E_tmp[i]-b-1));
			}
		}
	#else
		// Solve for the excitation, a PDE
		for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=my_n+2)
		{
			E_tmp = E + j;
			E_prev_tmp = E_prev + j;

			for (i = 0; i < my_n; i++)
			{
				E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(my_n+2)]+E_prev_tmp[i-(my_n+2)]);
			}
		}

		/* 
		 * Solve the ODE, advancing excitation and recovery variables
		 *     to the next timtestep
		 */

		for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(my_n+2))
		{
			E_tmp = E + j;
			R_tmp = R + j;
			for (i = 0; i < my_n; i++)
			{
				E_tmp[i] += -dt*(kk*E_tmp[i]*(E_tmp[i]-a)*(E_tmp[i]-1)+E_tmp[i]*R_tmp[i]);
				R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_tmp[i]+M2))*(-R_tmp[i]-kk*E_tmp[i]*(E_tmp[i]-b-1));
			}
		}
	#endif
		/////////////////////////////////////////////////////////////////////////////////

		if (cb.stats_freq)
		{
			if ( !(niter % cb.stats_freq))
			{
				stats(E, m, n, &mx, &sumSq);
				double l2norm = L2Norm(sumSq);
				repNorms(l2norm,mx,dt,m,n,niter, cb.stats_freq);
			}
		}

		if (cb.plot_freq)
		{
			if (!(niter % cb.plot_freq))
			{
				plotter->updatePlot(E,  niter, m, n);
			}
		}

		// Swap current and previous meshes
		double *tmp = E; E = E_prev; E_prev = tmp;

	} //end of 'niter' loop at the beginning

	// return the L2 and infinity norms via in-out parameters
	stats(E_prev,m,n,&Linf,&sumSq);
	L2 = L2Norm(sumSq);

	// Swap pointers so we can re-use the arrays
	*_E = E;
	*_E_prev = E_prev;
	
}
