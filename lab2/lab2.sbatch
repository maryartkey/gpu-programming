#!/bin/bash
#SBATCH -J cat 
#SBATCH -p compclass_unstable 
#SBATCH -e cat.%j.err
#SBATCH -o cat.%j.out
#SBATCH -N 1  
#SBATCH -n 3  
#SBATCH -c 4    
#SBATCH -t 00:01:00

module load nvidia/cuda 
                        
echo "Current path= pwd"
echo "node=hostname"
echo "nproc=nproc"
echo $SLURM_JOBID   
echo $SLURM_SUBMIT_DIR 
echo $SLURM_JOB_NODELIST
echo $SLURM_CPUS_PER_TASK
echo $SLURM_NTASKS  

nvcc lab2_fin.cu -o lab2 -lpng -lpng16 -Xcompiler -fopenmp
./lab2

sleep 10