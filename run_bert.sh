#!/bin/bash

################ 
# INITIALISE
################

abs_start=`date +%s`
printf "Process started at %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" > log.txt

source activate vineeths-tf

conda install -c nvidia cuda-nvcc --yes
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX

################ 
# BERT
################

start=`date +%s`

printf "Training BERT. Process started at %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> log.txt

printf "Training started at %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" > results/BERT/log.txt

python main.py --train-model --model-name BERT

printf "Evaluation started at %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> results/BERT/log.txt

python main.py --model-name BERT

end=`date +%s`
printf "Training and Evaluation Complete. Process ended at %s\n\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> results/BERT/log.txt
printf "Total time taken: %s" "$(date -u -d @$(($end-$start)) +'%H:%M:%S')" >> results/BERT/log.txt

printf "BERT Trained and Evaluated. Process ended at %s\n\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> log.txt

################ 
# END
################

abs_end=`date +%s`

printf "Complete! Process ended at %s\n\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> log.txt
printf "Total time taken: %s" "$(date -u -d @$(($abs_end-$abs_start)) +'%H:%M:%S')" >> log.txt
