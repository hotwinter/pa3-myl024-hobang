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

enum { NORTH = 0, EAST, WEST, SOUTH };

void repNorms(double l2norm, double mx, double dt, int m,int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);

extern control_block cb;

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

// Either pads the matrix with boundary condiditions (if the block lies on the problem boundaries)
// Or waits for the message that holds the needed ghost cells
void fillGhostCells(double *E, double *E_prev)
{
	MPI_Status status;

	int rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int n = cb.n;
	int m = cb.m;

	int x = rank % n;
	int y = rank % m;

	double* msg_N = new double[2*(m + n)];
	double* msg_E = msg_N + n;
	double* msg_W = msg_E + m;
	double* msg_S = msg_W + m;
	

	// 4 FOR LOOPS set up the padding needed for the boundary conditions
	int i,j;

	// Fills in the WEST Ghost Cells
	if (x == 0)
	{
		for (i = 0; i < (m+2)*(n+2); i+=(n+2))
		{
			E_prev[i] = E_prev[i+2];
		}
	}
	else
	{
		MPI_Recv(msg_W, m, MPI_DOUBLE, rank - 1, (rank - 1) << 2 + EAST, MPI_COMM_WORLD, &status);
	}

	// Fills in the EAST Ghost Cells
	if (x == n-1)
	{
		for (i = (n+1); i < (m+2)*(n+2); i+=(n+2))
		{
			E_prev[i] = E_prev[i-2];
		}
	}
	else
	{
		MPI_Recv(msg_E, m, MPI_DOUBLE, rank + 1, (rank + 1) << 2 + WEST, MPI_COMM_WORLD, &status);
	}

	// Fills in the NORTH Ghost Cells
	if (y == 0)
	{
		for (i = 0; i < (n+2); i++)
		{
			E_prev[i] = E_prev[i + (n+2)*2];
		}
	}
	else
	{
		//MPI_Recv(msg_N, n, MPI_DOUBLE, rank + )
	}

	// Fills in the SOUTH Ghost Cells
	if (y == n-1)
	{
		for (i = ((m+2)*(n+2)-(n+2)); i < (m+2)*(n+2); i++)
		{
			E_prev[i] = E_prev[i - (n+2)*2];
		}
	}
	else
	{

	}

	delete[] msg_N;
}


int solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter)
{
	// Simulated time is different from the integer timestep number
	double t = 0.0;
	
	double *E = *_E, *E_prev = *_E_prev;
	double *R_tmp = R;
	double *E_tmp = *_E;
	double *E_prev_tmp = *_E_prev;
	int niter;
	int m = cb.m, n=cb.n;
	int innerBlockRowStartIndex = (n+2)+1;
	int innerBlockRowEndIndex = (((m+2)*(n+2) - 1) - (n)) - (n+2);


	// We continue to sweep over the mesh until the simulation has reached
	// the desired number of iterations
	for (niter = 0; niter < cb.niters; niter++)
	{
		if (cb.debug && (niter==0))
		{
			double mx;
			double sumSq;
			stats(E_prev,m,n,&mx,&sumSq);
			double l2norm = L2Norm(sumSq);
			repNorms(l2norm,mx,dt,m,n,niter, cb.stats_freq);

			if (cb.plot_freq)
			{
				plotter->updatePlot(E,  niter, m+1, n+1);
			}
		}

		/* 
		 * Copy data from boundary of the computational box to the
		 * padding region, set up for differencing computational box's boundary
		 *
		 * These are physical boundary conditions, and are not to be confused
		 * with ghost cells that we would use in an MPI implementation
		 *
		 * The reason why we copy boundary conditions is to avoid
		 * computing single sided differences at the boundaries
		 * which increase the running time of solve()
		 *
		 */
 
		// 4 FOR LOOPS set up the padding needed for the boundary conditions
		int i,j;

		// Fills in the TOP Ghost Cells
		for (i = 0; i < (n+2); i++)
		{
			E_prev[i] = E_prev[i + (n+2)*2];
		}

		// Fills in the RIGHT Ghost Cells
		for (i = (n+1); i < (m+2)*(n+2); i+=(n+2))
		{
			E_prev[i] = E_prev[i-2];
		}

		// Fills in the LEFT Ghost Cells
		for (i = 0; i < (m+2)*(n+2); i+=(n+2))
		{
			E_prev[i] = E_prev[i+2];
		}

		// Fills in the BOTTOM Ghost Cells
		for (i = ((m+2)*(n+2)-(n+2)); i < (m+2)*(n+2); i++)
		{
			E_prev[i] = E_prev[i - (n+2)*2];
		}

		//////////////////////////////////////////////////////////////////////////////

		// #define FUSED 1

	#ifdef FUSED
		// Solve for the excitation, a PDE
		for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2))
		{
			E_tmp = E + j;
			E_prev_tmp = E_prev + j;
			R_tmp = R + j;

			for (i = 0; i < n; i++)
			{
				E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
				E_tmp[i] += -dt*(kk*E_tmp[i]*(E_tmp[i]-a)*(E_tmp[i]-1)+E_tmp[i]*R_tmp[i]);
				R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_tmp[i]+M2))*(-R_tmp[i]-kk*E_tmp[i]*(E_tmp[i]-b-1));
			}
		}
	#else
		// Solve for the excitation, a PDE
		for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2))
		{
			E_tmp = E + j;
			E_prev_tmp = E_prev + j;

			for (i = 0; i < n; i++)
			{
				E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
			}
		}

		/* 
		 * Solve the ODE, advancing excitation and recovery variables
		 *     to the next timtestep
		 */

		for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2))
		{
			E_tmp = E + j;
			R_tmp = R + j;
			for (i = 0; i < n; i++)
			{
				E_tmp[i] += -dt*(kk*E_tmp[i]*(E_tmp[i]-a)*(E_tmp[i]-1)+E_tmp[i]*R_tmp[i]);
				R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_tmp[i]+M2))*(-R_tmp[i]-kk*E_tmp[i]*(E_tmp[i]-b-1));
			}
		}
	#endif
		/////////////////////////////////////////////////////////////////////////////////

		if ((cb.stats_freq) && (niter !=0))
		{
			double mx;
			double sumSq;
			stats(E_prev,m,n,&mx,&sumSq);
			double l2norm = L2Norm(sumSq);
			repNorms(l2norm,mx,dt,m,n,niter, cb.stats_freq);
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

	// Swap pointers so we can re-use the arrays
	*_E = E;
	*_E_prev = E_prev;
	
	return niter;
}
