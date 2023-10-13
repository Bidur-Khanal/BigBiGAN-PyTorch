#! bin bash -l

dir="sbatch_log"
job_File="sbatch_run.sh" 
dataset=$"histopathology"

EXPT=BigGAN_mura
STD=$dir/STD_BigGAN_histopathology.out
ERR=$dir/ERR_BigGAN_histopathology.err
export dataset;
sbatch -J $EXPT -o $STD -t 03-18:00:00 -e $ERR $job_File
     

