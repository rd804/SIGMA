#!/bin/bash

#salloc --qos interactive --time=4:00:00 --nodes=1 --account=m4539 --gres=gpu:1 --constraint=gpu --image rd804/ranode_llf:latest --cpus-per-task=32 --ntasks-per-node=1
shifter --image=rd804/ranode_llf:latest jupyter notebook --port=8080 --no-browser

