#!/bin/bash

# discrete
# unscaled
# scaled

#############################################################
#############################################################

# non lin embedding experiments
# nsig=2000 1000
# x_train=data SR CR
# data= baseline and baseline with delta_r

# get all nodes
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
# node1=$(echo $nodes | cut -d ' ' -f 1)
# node2=$(echo $nodes | cut -d ' ' -f 2)
# node3=$(echo $nodes | cut -d ' ' -f 3)
# len of nodes
len=$(echo $nodes | wc -w)
echo "length of nodes: ${len}"

#tasks_per_node=3

tasks_per_node=$SLURM_NTASKS_PER_NODE
total_tasks=$SLURM_NTASKS

echo "Total tasks: ${total_tasks}"
echo "Tasks per node: ${tasks_per_node}"

epochs=2000
hidden_dim=256
#frequencies=4
batch_size=4096
num_blocks=4
wandb_group=debugging_nflow_interp

#n_sig_list=(2000 1000)
n_sig_list=(1000)
frequency_list=(8)
#x_train_list=(data SR CR no_signal)
#x_train_list=(data SR CR no_signal)
x_train_list=(data no_signal SR CR)
#x_train_list=(data)



local_task_counter=0
node_counter=0
task_counter=0


for frequencies in ${frequency_list[@]}; do
    for n_sig in ${n_sig_list[@]}; do
        for x_train in ${x_train_list[@]}; do

            wandb_job_type=freq_${frequencies}_nsig_${n_sig}
            data_dir=data/extended1
            data_name=base_scaled_mass
            
        
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

            srun --nodelist=${node} -n 1 -N 1 --exact --gpus-per-task=1 shifter python -u scripts/flow_matching_baseline.py --n_sig=${n_sig} \
                --epochs=${epochs} --batch_size=${batch_size} \
                --data_dir=${data_dir} --wandb_group=${wandb_group} --wandb_run_name=${data_name}_${x_train} \
                --hidden_dim=${hidden_dim} --wandb_job_type=${wandb_job_type} \
                --num_blocks=${num_blocks} --wandb --device=cuda:0 --non_linear_context --baseline --scaled_mass \
                --time_frequencies=3 --context_frequencies=${frequencies} --sample_interpolated \
                --x_train=${x_train} &>./results/${data_name}_${x_train}_${n_sig}.out &


        done
    done

    #n_sig_list=(2000 1000)
    #frequencies=4


    for n_sig in ${n_sig_list[@]}; do
        for x_train in ${x_train_list[@]}; do     

            wandb_job_type=freq_${frequencies}_nsig_${n_sig}
            data_dir=data/baseline_delta_R
            data_name=base_dR_scaled_mass
            

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
            echo "doing baseline delta R, task: ${n_sig} ${x_train} on node: ${node}"

            srun --nodelist=${node} -n 1 -N 1 --exact --gpus-per-task=1 shifter python -u scripts/flow_matching_baseline.py --n_sig=${n_sig} \
                --epochs=${epochs} --batch_size=${batch_size} \
                --data_dir=${data_dir} --wandb_group=${wandb_group} --wandb_run_name=${data_name}_${x_train} \
                --hidden_dim=${hidden_dim} --wandb_job_type=${wandb_job_type} \
                --num_blocks=${num_blocks} --wandb --device=cuda:0 --non_linear_context \
                --time_frequencies=3 --context_frequencies=${frequencies} --sample_interpolated --scaled_mass \
                --x_train=${x_train} &>./results/${data_name}_${x_train}_${n_sig}.out &

        done
    done
done

#############################################################
#############################################################


for frequencies in ${frequency_list[@]}; do
    for n_sig in ${n_sig_list[@]}; do
        for x_train in ${x_train_list[@]}; do

            wandb_job_type=freq_${frequencies}_nsig_${n_sig}
            data_dir=data/extended1
            data_name=base_unscaled_mass
            
        
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

            srun --nodelist=${node} -n 1 -N 1 --exact --gpus-per-task=1 shifter python -u scripts/flow_matching_baseline.py --n_sig=${n_sig} \
                --epochs=${epochs} --batch_size=${batch_size} \
                --data_dir=${data_dir} --wandb_group=${wandb_group} --wandb_run_name=${data_name}_${x_train} \
                --hidden_dim=${hidden_dim} --wandb_job_type=${wandb_job_type} \
                --num_blocks=${num_blocks} --wandb --device=cuda:0 --non_linear_context --baseline \
                --time_frequencies=3 --context_frequencies=${frequencies} --sample_interpolated \
                --x_train=${x_train} &>./results/${data_name}_${x_train}_${n_sig}.out &


        done
    done

    #n_sig_list=(2000 1000)
    #frequencies=4


    for n_sig in ${n_sig_list[@]}; do
        for x_train in ${x_train_list[@]}; do     

            wandb_job_type=freq_${frequencies}_nsig_${n_sig}
            data_dir=data/baseline_delta_R
            data_name=base_dR_unscaled_mass
            

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
            echo "doing baseline delta R, task: ${n_sig} ${x_train} on node: ${node}"

            srun --nodelist=${node} -n 1 -N 1 --exact --gpus-per-task=1 shifter python -u scripts/flow_matching_baseline.py --n_sig=${n_sig} \
                --epochs=${epochs} --batch_size=${batch_size} \
                --data_dir=${data_dir} --wandb_group=${wandb_group} --wandb_run_name=${data_name}_${x_train} \
                --hidden_dim=${hidden_dim} --wandb_job_type=${wandb_job_type} \
                --num_blocks=${num_blocks} --wandb --device=cuda:0 --non_linear_context \
                --time_frequencies=3 --context_frequencies=${frequencies} --sample_interpolated \
                --x_train=${x_train} &>./results/${data_name}_${x_train}_${n_sig}.out &

        done
    done
done