#!/bin/bash

################ 
# INITIALISE
################

abs_start=`date +%s`
printf "Process started at %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" > log.txt

source activate vineeths-nli

################################
# INTERVIEWED
################################

################ 
# BiGRU
################

start=`date +%s`

printf "Training BiGRU. Process started at %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> log.txt

printf "Training started at %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" > results/BiGRU/log.txt

python main.py --model-name BiGRU --data interviewed --wandb

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

python main.py --model-name SumEmbeddings --data interviewed --wandb

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

python main.py --model-name AvgEmbeddings --data interviewed --wandb

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

python main.py --model-name BiLSTM --data interviewed --wandb

end=`date +%s`
printf "Training and Evaluation Complete. Process ended at %s\n\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> results/BiLSTM/log.txt
printf "Total time taken: %s" "$(date -u -d @$(($end-$start)) +'%H:%M:%S')" >> results/BiLSTM/log.txt

printf "BiLSTM Trained and Evaluated. Process ended at %s\n\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> log.txt

################################
# HIRED
################################

################ 
# BiGRU
################

start=`date +%s`

printf "Training BiGRU. Process started at %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> log.txt

printf "Training started at %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" > results/BiGRU/log.txt

python main.py --model-name BiGRU --data hired --wandb

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

python main.py --model-name SumEmbeddings --data hired --wandb

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

python main.py --model-name AvgEmbeddings --data hired --wandb

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

python main.py --model-name BiLSTM --data hired --wandb

end=`date +%s`
printf "Training and Evaluation Complete. Process ended at %s\n\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> results/BiLSTM/log.txt
printf "Total time taken: %s" "$(date -u -d @$(($end-$start)) +'%H:%M:%S')" >> results/BiLSTM/log.txt

printf "BiLSTM Trained and Evaluated. Process ended at %s\n\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> log.txt

################ 
# END
################

abs_end=`date +%s`

printf "Complete! Process ended at %s\n\n" "$(date +'%Y-%m-%d %H:%M:%S')" >> log.txt
printf "Total time taken: %s" "$(date -u -d @$(($abs_end-$abs_start)) +'%H:%M:%S')" >> log.txt
