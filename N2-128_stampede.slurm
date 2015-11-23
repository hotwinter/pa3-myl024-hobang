#!/bin/bash
#SBATCH -A TG-CCR100030
#SBATCH -J APF-STRONG_N1
#SBATCH -o APF-STRONG_N1-%j.o
#SBATCH -e APF-STRONG_N1-%j.e
# Run on 4 (N) nodes with 16 (n/N) tasks per node
#SBATCH -n 128                           # total number of mpi tasks requested
#                                         not to exceed 16
#SBATCH -N 16                             # total number of nodes requested
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
ibrun -np 128 ./apf -n 400 -i 500 -x 2 -y 64
ibrun -np 128 ./apf -n 400 -i 500 -x 64 -y 2
ibrun -np 128 ./apf -n 400 -i 500 -x 4 -y 32 
ibrun -np 128 ./apf -n 400 -i 500 -x 32 -y 4
ibrun -np 128 ./apf -n 400 -i 500 -x 8 -y 16
ibrun -np 128 ./apf -n 400 -i 500 -x 16 -y 8

ibrun -np 128 ./apf -n 1800 -i 2000 -x 2 -y 64
ibrun -np 128 ./apf -n 1800 -i 2000 -x 64 -y 2
ibrun -np 128 ./apf -n 1800 -i 2000 -x 4 -y 32 
ibrun -np 128 ./apf -n 1800 -i 2000 -x 32 -y 4
ibrun -np 128 ./apf -n 1800 -i 2000 -x 8 -y 16
ibrun -np 128 ./apf -n 1800 -i 2000 -x 16 -y 8

ibrun -np 128 ./apf -n 8000 -i 2000 -x 2 -y 64
ibrun -np 128 ./apf -n 8000 -i 2000 -x 64 -y 2
ibrun -np 128 ./apf -n 8000 -i 2000 -x 4 -y 32 
ibrun -np 128 ./apf -n 8000 -i 2000 -x 32 -y 4
ibrun -np 128 ./apf -n 8000 -i 2000 -x 8 -y 16
ibrun -np 128 ./apf -n 8000 -i 2000 -x 16 -y 8

date
echo ">>> Job Ends"
