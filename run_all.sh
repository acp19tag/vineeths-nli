#!/bin/bash

################ 
# INITIALISE
################

abs_start=`date +%s`
printf "Process started at %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" > log.txt

source activate vineeths-nli

################ 
# BiGRU
################

start=`date +%s`

printf "Training BiGRU. Process started at %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> log.txt

printf "Training started at %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" > results/BiGRU/log.txt

python main.py --train-model --model-name BiGRU

printf "Evaluation started at %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> results/BiGRU/log.txt

python main.py --model-name BiGRU


end=`date +%s`
printf "Training and Evaluation Complete. Process ended at %s\n\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> results/BiGRU/log.txt
printf "Total time taken: %s" "$(date -u -d @$(($end-$start)) +'%H:%M:%S')" >> results/BiGRU/log.txt

printf "BiGRU Trained and Evaluated. Process ended at %s\n\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> log.txt

################ 
# SumEmbeddings
################

start=`date +%s`

printf "Training SumEmbeddings. Process started at %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> log.txt

printf "Training started at %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" > results/SumEmbeddings/log.txt

python main.py --train-model --model-name SumEmbeddings

printf "Evaluation started at %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> results/SumEmbeddings/log.txt

python main.py --model-name SumEmbeddings

end=`date +%s`
printf "Training and Evaluation Complete. Process ended at %s\n\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> results/SumEmbeddings/log.txt
printf "Total time taken: %s" "$(date -u -d @$(($end-$start)) +'%H:%M:%S')" >> results/SumEmbeddings/log.txt

printf "SumEmbeddings Trained and Evaluated. Process ended at %s\n\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> log.txt

################ 
# AvgEmbeddings
################

start=`date +%s`

printf "Training AvgEmbeddings. Process started at %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> log.txt

printf "Training started at %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" > results/AvgEmbeddings/log.txt

python main.py --train-model --model-name AvgEmbeddings

printf "Evaluation started at %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> results/AvgEmbeddings/log.txt

python main.py --model-name AvgEmbeddings

end=`date +%s`
printf "Training and Evaluation Complete. Process ended at %s\n\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> results/AvgEmbeddings/log.txt
printf "Total time taken: %s" "$(date -u -d @$(($end-$start)) +'%H:%M:%S')" >> results/AvgEmbeddings/log.txt

printf "AvgEmbeddings Trained and Evaluated. Process ended at %s\n\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> log.txt

################ 
# BiLSTM
################

start=`date +%s`

printf "Training BiLSTM. Process started at %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> log.txt

printf "Training started at %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" > results/BiLSTM/log.txt

python main.py --train-model --model-name BiLSTM

printf "Evaluation started at %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> results/BiLSTM/log.txt

python main.py --model-name BiLSTM

end=`date +%s`
printf "Training and Evaluation Complete. Process ended at %s\n\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> results/BiLSTM/log.txt
printf "Total time taken: %s" "$(date -u -d @$(($end-$start)) +'%H:%M:%S')" >> results/BiLSTM/log.txt

printf "BiLSTM Trained and Evaluated. Process ended at %s\n\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> log.txt

################ 
# BERT
################

conda install -c nvidia cuda-nvcc --yes
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX

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
