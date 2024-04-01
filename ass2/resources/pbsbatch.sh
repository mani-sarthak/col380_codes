#!/bin/sh
### Set the job name (for your reference)
#PBS -N check
### Set the project name, your department code by default
#PBS -P col380.cs1210095.course
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in
####
#PBS -l select=2:ncpus=1:ngpus=2
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=01:00:00

#PBS -l software=nvcc
module load compiler/cuda/10.2/compilervars

# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR
#job 
time -p mpirun ./cuda