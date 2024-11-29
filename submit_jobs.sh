#!/bin/bash

#salloc --qos shared_interactive --time=4:00:00 --nodes=1 --account=m4539 --gres=gpu:1 --constraint=gpu --image rd804/ranode_llf:latest --ntasks-per-node=1 --cpus-per-task=32 
#shifter --image=rd804/ranode_llf:latest jupyter notebook --port=1234 --no-browser

# Request an allocation using salloc
#salloc --time=01:00:00 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=4G
salloc --qos interactive --time=4:00:00 --nodes=4 --account=m4539 --gres=gpu:4 --constraint=gpu --image rd804/ranode_llf:latest --ntasks-per-node=4 bash parallel_script.sh 0
salloc --qos interactive --time=4:00:00 --nodes=4 --account=m4539 --gres=gpu:4 --constraint=gpu --image rd804/ranode_llf:latest --ntasks-per-node=4 bash parallel_script.sh 64



