#!/bin/bash

Lx=$1
Ly=$2
D=$3
J=$4
L=$5
Sz=$6
N=$7
finished_params_filename=$8

mkdir "parameters_${D}_${J}_${L}"
cd "parameters_${D}_${J}_${L}"
finished_params_filename="../$finished_params_filename"

NAME=`echo "cylinder_DMRG_${Lx}_${Ly}_${D}_${J}_${L}_${Sz}"`

PBS="#!/bin/bash\n\
#PBS -N ${NAME}\n\
#PBS -l walltime=01:00:00\n\
#PBS -l select=1:ncpus=8:mem=500MB\n\
#PBS -l software=one_sector_dmrg.py\n\
#PBS -m n\n\
cd \$PBS_O_WORKDIR\n\
python3 -W ignore ../one_sector_dmrg.py $Lx $Ly $D $J $L $Sz $N $finished_params_filename >& cylinder_DMRG_${Lx}_${Ly}_${D}_${J}_${L}_${Sz}.txt"

# Echo the string PBS to the function qsub, which submits it as a cluster job for you
# A small delay is included to avoid overloading the submission process

echo -e ${PBS} | qsub
#echo %{$PBS}
sleep 0.5
echo "done."
