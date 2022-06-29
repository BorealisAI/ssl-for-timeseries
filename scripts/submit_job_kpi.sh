#!/bin/bash
# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
for seed in 1 2 3 4 5
do
    echo "#!/bin/bash" > jobscript.sh
    echo "#SBATCH --time=7-00:00:00" >> jobscript.sh
    echo "#SBATCH --mem=10G" >> jobscript.sh
    echo "#SBATCH --gres=gpu:1" >> jobscript.sh
    echo "#SBATCH --nodelist=${1}" >> jobscript.sh
    echo "#SBATCH --job-name=${2}_seed_${seed}" >> jobscript.sh
    echo "#SBATCH --cpus-per-task=4" >> jobscript.sh
    echo "#SBATCH --ntasks=1" >> jobscript.sh
    echo "#SBATCH --output=output/%x-%j.out" >> jobscript.sh

    echo "source activate atom-ssl-ts" >> jobscript.sh
    echo 'echo "$(date): job $SLURM_JOBID starting on $SLURM_NODELIST"' >> jobscript.sh
    echo "hostname" >> jobscript.sh
    echo "whoami" >> jobscript.sh
    echo "nvidia-smi" >> jobscript.sh
    echo 'echo "CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES)"' >> jobscript.sh
    echo "# Print information about the branch and commit used." >> jobscript.sh
    echo 'echo -e "\nGit"' >> jobscript.sh
    echo 'echo -e "$(git branch -vv)"' >> jobscript.sh
    echo 'echo -e "current commit: $(git log --pretty=format:'%h' -n 1)  $(git log --pretty=format:'%H' -n 1)"' >> jobscript.sh

    echo "python -m ssl.run \
        --dataset kpi \
        --run-name anomaly \
        --loader anomaly \
        --repr-dims 320 \
        --max-threads 8 \
        --seed ${seed} \
        --queue-size 128 \
        --max-seq-len 3108 \
        --alpha 0.5 \
        --similarity INSTANCE TEMPORAL \
        --hierarchical False \
        --temperature 0.07 \
        --eval" >> jobscript.sh

    #submit the job
    sbatch jobscript.sh

    # remove the created script after submitting
    rm jobscript.sh
done
