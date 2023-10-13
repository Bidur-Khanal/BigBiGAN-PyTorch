#! bin bash -l

dir="sbatch_log"
job_File="sbatch_run.sh" 
dataset=$"mura"

EXPT=BigGAN_mura
STD=$dir/STD_BigGAN_mura.out
ERR=$dir/ERR_BigGAN_mura.err
export dataset;
sbatch -J $EXPT -o $STD -t 00-18:00:00 -e $ERR $job_File
     

