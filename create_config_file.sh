#!/bin/bash

output_file="signal_scan_config.txt"

n_sig=(3000 2000 1500 1000 750 500 450 300)
tries=(0 1 2 3 4 5 6 7 8 9)

printf "%-15s %-15s %-15s\n" "ArrayTaskID" "n_sig" "tries" > "$output_file"

i=0
for n in ${n_sig[@]}; do
    for t in ${tries[@]}; do
        printf "%-15s %-15s %-15s\n" $i $n $t >> "$output_file"
        i=$((i+1))
        #done
    done    
done    

