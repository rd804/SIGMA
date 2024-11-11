#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --time=1:00:00
#SBATCH --constraint=gpu 
#SBATCH --image=rd804/ranode_llf:latest
#SBATCH --account=m4539
#SBATCH --qos=shared
#SBATCH --requeue 
#SBATCH --gres=gpu:1
#SBATCH --array=0-79
#SBATCH --job-name=fm_interp
#SBATCH --output=slurm/output/fm_ranode_scan_blocks_%a.out
#SBATCH --error=slurm/error/fm_ranode_scan_blocks_%a.err


epochs=2
hidden_dim=256
batch_size=4096
num_blocks=4
wandb_group=signal_scan
frequencies=9
data_dir=data/baseline_delta_R
x_train=data
          
config=signal_scan_config.txt

n_sig=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
seed=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

wandb_run_name=seed_${seed}
wandb_job_type=nsig_${n_sig}

shifter python -u scripts/flow_matching_baseline.py --n_sig=${n_sig} \
                --epochs=${epochs} --batch_size=${batch_size} \
                --data_dir=${data_dir} --wandb_group=${wandb_group} --wandb_run_name=${wandb_run_name} \
                --hidden_dim=${hidden_dim} --wandb_job_type=${wandb_job_type} \
                --num_blocks=${num_blocks} --wandb --device=cuda:0 --non_linear_context \
                --time_frequencies=3 --context_frequencies=${frequencies} --sample_interpolated --scaled_mass \
                --x_train=${x_train} --seed=${seed} --resample

# shifter python -u scripts/flow_matching_baseline.py --n_sig=${n_sig} \
#                 --epochs=${epochs} --batch_size=${batch_size} \
#                 --data_dir=${data_dir} --wandb_group=${wandb_group} --wandb_run_name=${wandb_run_name} \
#                 --hidden_dim=${hidden_dim} --wandb_job_type=${wandb_job_type} \
#                 --num_blocks=${num_blocks} --wandb --device=cuda:0 --non_linear_context --baseline --scaled_mass \
#                 --time_frequencies=3 --context_frequencies=${frequencies} --sample_interpolated \
#                 --x_train=${x_train} --seed=${seed} --resample 