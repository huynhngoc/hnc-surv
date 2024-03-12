#!/bin/bash
#SBATCH --ntasks=1               # 1 core(CPU)
#SBATCH --nodes=1                # Use 1 node
#SBATCH --job-name=interpret   # sensible name for the job
#SBATCH --mem=12G                 # Default memory per CPU is 3GB.
#SBATCH --partition=smallmem,hugemem,orion,hugemem-avx2 # Use the verysmallmem-partition for jobs requiring < 10 GB RAM.
#SBATCH --mail-user=ngochuyn@nmbu.no # Email me when job is done.
#SBATCH --mail-type=FAIL
#SBATCH --output=outputs/interpret-analyze-%A-%a.out
#SBATCH --error=outputs/interpret-analyze-%A-%a.out

# If you would like to use more please adjust this.

## Below you can put your scripts
# If you want to load module
module load singularity

## Code
# If data files aren't copied, do so
#!/bin/bash
if [ $# -lt 2 ];
    then
    printf "Not enough arguments - %d\n" $#
    exit 0
    fi

# if [ ! -d "$PROJECTS/ngoc/hn_surv/perf/$2/OUS/raw" ]
#     then
#     echo "Didn't find OUS raw result folder. Creating folder..."
#     mkdir --parents $PROJECTS/ngoc/hn_surv/perf/$2/OUS/raw
#     fi

# if [ ! -d "$PROJECTS/ngoc/hn_surv/perf/$2/OUS/smoothen" ]
#     then
#     echo "Didn't find OUS smoothen result folder. Creating folder..."
#     mkdir --parents $PROJECTS/ngoc/hn_surv/perf/$2/OUS/smoothen
#     fi
if [ ! -d "$PROJECTS/ngoc/hn_surv/perf/$2/OUS/smoothen_v2" ]
    then
    echo "Didn't find OUS smoothen result folder. Creating folder..."
    mkdir --parents $PROJECTS/ngoc/hn_surv/perf/$2/OUS/smoothen_v2
    fi

# if [ ! -d "$PROJECTS/ngoc/hn_surv/perf/$2/MAASTRO/raw" ]
#     then
#     echo "Didn't find MAASTRO raw result folder. Creating folder..."
#     mkdir --parents $PROJECTS/ngoc/hn_surv/perf/$2/MAASTRO/raw
#     fi

# if [ ! -d "$PROJECTS/ngoc/hn_surv/perf/$2/MAASTRO/smoothen" ]
#     then
#     echo "Didn't find MAASTRO smoothen result folder. Creating folder..."
#     mkdir --parents $PROJECTS/ngoc/hn_surv/perf/$2/MAASTRO/smoothen
#     fi

if [ ! -d "$PROJECTS/ngoc/hn_surv/perf/$2/MAASTRO/smoothen_v2" ]
    then
    echo "Didn't find MAASTRO smoothen result folder. Creating folder..."
    mkdir --parents $PROJECTS/ngoc/hn_surv/perf/$2/MAASTRO/smoothen_v2
    fi

echo "Finished seting up files."


# Run experiment
# export ITER_PER_EPOCH=200
export NUM_CPUS=4
export RAY_ROOT=$TMPDIR/ray
export MAX_SAVE_STEP_GB=0
singularity exec --nv deoxys-survival.sif python -u interpret_analyze.py $1 $PROJECTS/ngoc/hn_surv/perf/$2 --idx $SLURM_ARRAY_TASK_ID

# echo "Finished training. Post-processing results"

# singularity exec --nv deoxys.sif python -u post_processing.py /net/fs-1/Ngoc/hnperf/$2 --temp_folder $SCRATCH/hnperf/$2 --analysis_folder $SCRATCH/analysis/$2 ${@:4}

# echo "Finished post-precessing. Running test on best model"

# singularity exec --nv deoxys.sif python -u run_test.py /net/fs-1/Ngoc/hnperf/$2 --temp_folder $SCRATCH/hnperf/$2 --analysis_folder $SCRATCH/analysis/$2 ${@:4}
