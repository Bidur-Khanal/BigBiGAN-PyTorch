#! bin bash -l

dir="sbatch_log"
job_File="sbatch_run.sh" 
dataset=$"dermnet"

EXPT=BigGAN_mura
STD=$dir/STD_BigGAN_dermnet.out
ERR=$dir/ERR_BigGAN_dermnet.err
export dataset;
sbatch -J $EXPT -o $STD -t 00-18:00:00 -e $ERR $job_File
     

