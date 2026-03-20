In order to perform the scan sufficiently, we suggest to run the code in a cluster. For example, we set up the subscripts of 125*4 cores  in HIAS cluster with the following code
------
#!/bin/bash
#PBS -N opt_c1
#PBS -l nodes=4:ppn=125
#PBS -l walltime=50:00:00
#PBS -o outfiles/optimization_c1_out.txt
#PBS -e errfiles/optimization_c1_err.txt

cd $PBS_O_WORKDIR
------

Run the following subscripts to perform the scan
------
echo "=== Starting time: $(date) ==="

mpirun -np 500 -machinefile $PBS_NODEFILE python optimize.py --model M1 --octant 2nd --n-points 5000 --n-generations 50000 --output-dir ./Parameters

echo "=== Ending time: $(date) ==="
------