#!/bin/bash
#SBATCH -J cudaarr  	# Название задачи (по Вашему усмотрению)
#SBATCH -p compclass 	# Название очереди (либо «a100serv», либо «compclass», либо «compclass_unstable»)
#SBATCH -e myjob.%j.err # название файла с потоком ошибок stderr (%j заменится на номер задачи)
#SBATCH -o myjob.%j.out # название файла с потоком вывода stdout (%j заменится на номер задачи)
#SBATCH -N 1  	# Количество требуемых для задачи вычислительных узлов (для очередей “gpuserv” # и «a100serv» может быть только «1»!)
#SBATCH -n 3  	# Количество требуемых MPI-процессов
#SBATCH -c 4  		# Количество процессорных ядер на задачу
#SBATCH -t 00:01:00  # Требуемое для задачи время работы в формате hh:mm:ss – в
                                           # данном случае 1 минута

module load nvidia/cuda  # Если задача не считает на графических ускорителях, то данную строку
                                                   # можно удалить
echo "Current path= `pwd`"
echo "node=`hostname`"
echo "nproc=`nproc`"
echo $SLURM_JOBID   		# Номер, который будет присвоен задаче
echo $SLURM_SUBMIT_DIR  		# Текущая директория для задачи
echo $SLURM_JOB_NODELIST 	# Узлы, выделенные для задачи
echo $SLURM_CPUS_PER_TASK  	# Количество процессорных ядер на MPI-процесс
echo $SLURM_NTASKS		# Количество MPI-процессов
sleep 10

nvcc -o lab1 lab1.cu
./lab1
