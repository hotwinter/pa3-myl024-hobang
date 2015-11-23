#!/bin/bash
#SBATCH -A TG-CCR100030
#SBATCH -J APF-STRONG_N1
#SBATCH -o APF-STRONG_N1-%j.o
#SBATCH -e APF-STRONG_N1-%j.e
# Run on 4 (N) nodes with 16 (n/N) tasks per node
#SBATCH -n 64                           # total number of mpi tasks requested
#                                         not to exceed 16
#SBATCH -N 4                             # total number of nodes requested
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
ibrun -np 16 ./apf -n 1800 -i 2000 -x 2 -y 8
ibrun -np 16 ./apf -n 1800 -i 2000 -x 4 -y 4
ibrun -np 16 ./apf -n 1800 -i 2000 -x 4 -y 4 -k
ibrun -np 32 ./apf -n 1800 -i 2000 -x 4 -y 8 
ibrun -np 32 ./apf -n 1800 -i 2000 -x 4 -y 8  -k
ibrun -np 32 ./apf -n 1800 -i 2000 -x 8 -y 4
ibrun -np 64 ./apf -n 1800 -i 2000 -x 4 -y 16
ibrun -np 64 ./apf -n 1800 -i 2000 -x 8 -y 8
ibrun -np 64 ./apf -n 1800 -i 2000 -x 8 -y 8 -k
date
echo ">>> Job Ends"
