#!/bin/bash
#SBATCH --ntasks=1               # 1 core(CPU)
#SBATCH --nodes=1                # Use 1 node
#SBATCH --job-name=head_neck   # sensible name for the job
#SBATCH --mem=200G                 # Default memory per CPU is 3GB.
#SBATCH --partition=gpu # Use the verysmallmem-partition for jobs requiring < 10 GB RAM.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=torjus.strandenes.moen@nmbu.no # Email me when job is done.
#SBATCH --mail-type=FAIL
#SBATCH --output=outputs/surv-%A.out
#SBATCH --error=outputs/surv-%A.out

# If you would like to use more please adjust this.

## Below you can put your scripts
# If you want to load module
module load singularity

## Code
# If data files aren't copied, do so
#!/bin/bash
if [ $# -lt 3 ];
    then
    printf "Not enough arguments - %d\n" $#
    exit 0
    fi

# if [ ! -d "$TMPDIR/$USER/hn_delin" ]
#     then
#     echo "Didn't find dataset folder. Copying files..."
#     mkdir --parents $TMPDIR/$USER/hn_delin
#     fi

# for f in $(ls $PROJECTS/ngoc/datasets/headneck/*)
#     do
#     FILENAME=`echo $f | awk -F/ '{print $NF}'`
#     echo $FILENAME
#     if [ ! -f "$TMPDIR/$USER/hn_delin/$FILENAME" ]
#         then
#         echo "copying $f"
#         cp -r $PROJECTS/ngoc/datasets/headneck/$FILENAME $TMPDIR/$USER/hn_delin/
#         fi
#     done


echo "Finished seting up files."

# Hack to ensure that the GPUs work
nvidia-modprobe -u -c=0

# Run experiment
# export ITER_PER_EPOCH=200
export MAX_SAVE_STEP_GB=0
export NUM_CPUS=4
export RAY_ROOT=$TMPDIR/ray
singularity exec --nv deoxys-survival.sif python experiment_survival.py $1 $PROJECTS/ngoc/hn_surv/perf/$2 --temp_folder $SCRATCH_PROJECTS/ceheads/hn_surv/$2 --analysis_folder $SCRATCH_PROJECTS/ceheads/hn_surv_analysis/$2 --epochs $3 ${@:4}

# echo "Finished training. Post-processing results"

# singularity exec --nv deoxys.sif python -u post_processing.py /net/fs-1/Ngoc/hnperf/$2 --temp_folder $SCRATCH/hnperf/$2 --analysis_folder $SCRATCH/analysis/$2 ${@:4}

# echo "Finished post-precessing. Running test on best model"

# singularity exec --nv deoxys.sif python -u run_test.py /net/fs-1/Ngoc/hnperf/$2 --temp_folder $SCRATCH/hnperf/$2 --analysis_folder $SCRATCH/analysis/$2 ${@:4}
