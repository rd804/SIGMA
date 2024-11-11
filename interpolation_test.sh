#!/bin/bash

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
TMOUT=180000
len=$(echo $nodes | wc -w)
echo "number of nodes: ${len}"


tasks_per_node=$SLURM_NTASKS_PER_NODE
total_tasks=$SLURM_NTASKS

echo "Total tasks: ${total_tasks}"
echo "Tasks per node: ${tasks_per_node}"
echo "nodes: ${nodes}"

epochs=2000
hidden_dim=256
batch_size=4096
num_blocks=4
wandb_group=no_freq
frequencies=9
data_dir=data/baseline_delta_R
x_train=data


# for interp_blocks in 0 1 2 3 4; do
#     echo "interp_blocks: ${interp_blocks}"
#     #interp_blocks=4
# for n_sig in 300 450 500 750 1000 1500 2000 3000; do
#     echo "n_sig: ${n_sig}"
#     for seed in 0 1 2 3 4 5 6 7 8 9; do
seed=1

for seed in 0 1 2 3 4 5 6 7 8 9; do
    echo "seed: ${seed}"
    n_sig=1000
    #echo "seed: ${seed}"
    #seed=8
    wandb_run_name=seed_${seed}
    wandb_job_type=nsig_${n_sig}

    srun --nodelist=${nodes} -n 1 -N 1 --exact --gpus-per-task=1 shifter python -u scripts/compare_interpolation_methods_no_embedding.py --n_sig=${n_sig} \
        --epochs=${epochs} --batch_size=${batch_size} \
        --data_dir=${data_dir} --wandb_group=${wandb_group} --wandb_run_name=${wandb_run_name} \
        --hidden_dim=${hidden_dim} --wandb_job_type=${wandb_job_type} \
        --num_blocks=${num_blocks} --device=cuda:0  \
        --time_frequencies=3  --context_frequencies=${frequencies} --seed=${seed} --resample \
        --non_linear_context --higher_mass=3.8 --lower_mass=3.2 --interpolation_method='vector' --x_train=${x_train} 
        # --interp_block=${interp_blocks} --scaled_mass
#     done
done

    # if [ $i -eq $task_end ]; then
    #     echo "Waiting for final set of tasks to finish"
    #     wait
    #     exit
    # fi


#done



