#!/bin/bash
#SBATCH -A TG-CCR100030
#SBATCH -J APF-STRONG_N2
#SBATCH -o APF-STRONG_N2-%j.o
#SBATCH -e APF-STRONG_N2-%j.e
# Run on 32 (N) nodes with 16 (n/N) tasks per node
#SBATCH -n 512                           # total number of mpi tasks requested
#                                          not to exceed 16
#SBATCH -N 32                            # total number of nodes requested
#SBATCH -p normal                        # queue (partition)
#SBATCH -t 00:02:00

#SBATCH --mail-user=youremail@ucsd.edu   # notification email
#SBATCH --mail-type=END,FAIL             # notifications for job done & fail

echo
echo " *** Current working directory"
pwd
echo
echo " *** Environment"
printenv

echo

echo ">>> Job Starts"
date
ibrun -np 512 ./apf-ref -n 8000 -i 2000 -x 32 -y 16
ibrun -np 512 ./apf-ref -n 8000 -i 2000 -x 16 -y 32
ibrun -np 512 ./apf-ref -n 8000 -i 2000 -x 16 -y 32 -k
date
echo ">>> Job Ends"
