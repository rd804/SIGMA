#!/bin/bash

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
TMOUT=180000
len=$(echo $nodes | wc -w)
echo "number of nodes: ${len}"


tasks_per_node=$SLURM_NTASKS_PER_NODE
total_tasks=$SLURM_NTASKS

echo "Total tasks: ${total_tasks}"
echo "Tasks per node: ${tasks_per_node}"


epochs=2000
hidden_dim=256
batch_size=4096
num_blocks=4
wandb_group=signal_scan
frequencies=9
data_dir=data/baseline_delta_R
x_train=data

# epochs=2000
# hidden_dim=256
# frequencies=3
# batch_size=4096
# num_blocks=4
# data_dir=data/baseline_delta_R
# #data_dir=data/extended1
# x_train=CR
wandb_group=signal_scan_baseline

#config=signal_scan_config.txt

task_start=$1
task_end=$((task_start+63))
#max_tasks=64

echo "task_start: ${task_start}"
echo "task_end: ${task_end}"

config=signal_scan_config.txt

local_task_counter=0
node_counter=0
task_counter=0
#max_task_counter=0

task_array=($(seq ${task_start} ${task_end}))

for i in ${task_array[@]}; do

    echo "task_id: ${i}"
    n_sig=$(awk -v ArrayTaskID=$i '$1==ArrayTaskID {print $2}' $config)
    seed=$(awk -v ArrayTaskID=$i '$1==ArrayTaskID {print $3}' $config)

    wandb_run_name=seed_${seed}
    wandb_job_type=nsig_${n_sig}

    if [ $local_task_counter -eq $tasks_per_node ]; then
        node_counter=$((node_counter+1))
        local_task_counter=1
    else
        local_task_counter=$((local_task_counter+1))
    fi

    if [ $task_counter -eq $total_tasks ]; then
        echo "Waiting for tasks to finish"
        wait
        task_counter=1
        node_counter=0
    else
        task_counter=$((task_counter+1))
    fi
    node=$(echo $nodes | cut -d ' ' -f $((node_counter+1)))
    echo "Node counter: ${node_counter}"
    echo "Node: ${node}"
    echo "Task: ${local_task_counter}"
    echo "doing baseline task: ${n_sig} ${x_train} on node: ${node}"
    echo "n_sig: ${n_sig}"
    echo "seed: ${seed}"


    srun --nodelist=${node} -n 1 -N 1 --exact --gpus-per-task=1 shifter python -u scripts/flow_matching_baseline.py --n_sig=${n_sig} \
        --epochs=${epochs} --batch_size=${batch_size} \
        --data_dir=${data_dir} --wandb_group=${wandb_group} --wandb_run_name=${wandb_run_name} \
        --hidden_dim=${hidden_dim} --wandb_job_type=${wandb_job_type} \
        --num_blocks=${num_blocks} --wandb --device=cuda:0  \
        --time_frequencies=3  --context_frequencies=${frequencies} --seed=${seed} --resample --baseline \
        --non_linear_context --sample_interpolated --scaled_mass \
        --x_train=${x_train} &>./results/${wandb_run_name}_${n_sig}.out &

    if [ $i -eq $task_end ]; then
        echo "Waiting for final set of tasks to finish"
        wait
        exit
    fi


done



