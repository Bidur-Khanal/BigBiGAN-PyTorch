#! bin bash -l

dir="sbatch_log"
job_File="sbatch_run.sh" 
dataset=$"covid19_xray"

EXPT=BigGAN_mura
STD=$dir/STD_BigGAN_covid19_xray.out
ERR=$dir/ERR_BigGAN_covid19_xray.err
export dataset;
sbatch -J $EXPT -o $STD -t 03-18:00:00 -e $ERR $job_File
     

