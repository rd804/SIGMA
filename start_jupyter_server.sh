
salloc --qos preempt --time=6:00:00 --nodes=1 --account=m4539 --gres=gpu:1 --constraint=gpu --image rd804/ranode_llf:latest --ntasks-per-node=1
#shifter --image=rd804/ranode_llf:latest jupyter notebook --port=8888 --no-browser

