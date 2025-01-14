# Imports
# sourcery skip: dict-assign-update-to-union
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import argparse
from utils.read_data import read_data
# from utils.generate_meta_input import generate_meta_input
from TFIDF.logistic_regression_train import logistic_regression_train
from TFIDF.logistic_regression_test import logistic_regression_test
from deep_model.SumEmbeddings.model_train import SE_model_train
from deep_model.SumEmbeddings.model_test import SE_model_test
from deep_model.AvgEmbeddings.model_train import AE_model_train
from deep_model.AvgEmbeddings.model_test import AE_model_test
from deep_model.BiLSTM.model_train import BL_model_train
from deep_model.BiLSTM.model_test import BL_model_test
from deep_model.BiGRU.model_train import BG_model_train
from deep_model.BiGRU.model_test import BG_model_test
from deep_model.BERT.model_train import BERT_model_train
from deep_model.BERT.model_test import BERT_model_test
import wandb

"""
# To be run once to generate cleaned and tokenized sentences stored as pickle files.
# Takes a significant amount of time to execute
"""
# generate_meta_input() 


# Command line argument parser. Defaults to testing the SumEmbeddings model.
arg_parser = argparse.ArgumentParser(description="Choose between training the model or testing the model. "
                                                 "Choose the model architecture between SumEmbeddings, "
                                                 "AvgEmbeddings, BiLSTM, BiGRU and BERT")

arg_parser.add_argument("--model-name", type=str, default="SumEmbeddings")
arg_parser.add_argument("--data", type = str, default='interviewed')
arg_parser.add_argument('--wandb', action='store_true')

argObj = arg_parser.parse_args()

model_name = argObj.model_name

# Reads the data from from pickle files
data_dir = f'../data/matched/reformatted/{argObj.data}'

data = read_data(data_dir)
train_data = data[:3]
test_data = data[3:]

if argObj.wandb:
    sweep_configuration = {
        'method': 'bayes',
        'name': 'vineeths-sweep',
        'metric': {'goal': 'maximize', 'name': 'dev_acc'},
        # 'early_terminate': {'type': 'hyperband', 'min_iter': 10},
        'parameters':
            {
                'batch_size': {
                    'values': [184, 256, 328]
                    # 'distribution': 'q_log_uniform_values',
                    # 'q': 8,
                    # 'min': 32,
                    # 'max': 1024,
                    },
                'seq_length': {
                    # 'values': [128, 256, 512]
                    'value': 128
                    },
                'learning_rate': {
                    'values': [1e-3, 1e-4, 1e-5]
                    # 'value': 1e-3
                    },
                'dropout': {
                    'values': [0.3, 0.5, 0.7]
                    # 'distribution': 'uniform',
                    # 'min': 0.0,
                    # 'max': 0.8,
                    # 'value': 0.3
                    },
                'num_epochs': {'value': 200},
                'embedding_size': {'value': 300},
                'vocab_size': {'value': 20000},
                'architecture': {'value': f'{model_name}'},
                'dataset': {'value': f"TribePad: {argObj.data}"}
            }
    }
    # STATIC config
    static_config = {
        'rho': 0.9,
        'epsilon': 1e-08,
        'decay': 0.0,
        'validation_split': 0.02,
        'activation': 'relu',
        'l2': 4e-6,
        'gru_units': 64,
        'lstm_units': 64
    }

    # if model_name in {"SumEmbeddings", "AvgEmbeddings", "BiGRU", "BiLSTM"}:
    #     sweep_configuration.update({
    #         'rho': {'value': 0.9},
    #         'epsilon': {'value': 1e-08},
    #         'decay': {'value': 0.0},
    #         'validation_split': {'value': 0.02},
    #     })
    # if model_name in {"BiGRU", "BiLSTM"}:
    #     sweep_configuration.update({
    #         'activation': {'value': 'relu'},
    #         'l2': {'value': 4e-6},
    #     })
    # if model_name == "BiGRU":
    #     sweep_configuration['gru_units'] = {'values': [64, 128, 256]}
    # if model_name == "BiLSTM":
    #     sweep_configuration['lstm_units'] = {'values': [64, 128, 256]}

    sweep_id = wandb.sweep(
        sweep_configuration,
        project = 'job-status-prediction'
    )

# warm up the nextwork
logistic_regression_train(train_data)

def main(config = None):
        
        if argObj.wandb: 
            with wandb.init(config = config):

                # TRAIN
                if model_name == "SumEmbeddings":
                    SE_model_train(train_data, wandb, static_config)
                elif model_name == "AvgEmbeddings":
                    AE_model_train(train_data, wandb, static_config)
                elif model_name == "BiGRU":
                    BG_model_train(train_data, wandb, static_config)
                elif model_name == "BiLSTM":
                    BL_model_train(train_data, wandb, static_config)
                elif model_name == "BERT":
                    BERT_model_train(train_data, wandb, static_config)

                # EVALUATE
                if model_name == "SumEmbeddings":
                    SE_model_test(test_data, wandb)
                elif model_name == "AvgEmbeddings":
                    AE_model_test(test_data, wandb)
                elif model_name == "BiGRU":
                    BG_model_test(test_data, wandb)
                elif model_name == "BiLSTM":
                    BL_model_test(test_data, wandb)
                elif model_name == "BERT":
                    BERT_model_test(test_data, wandb)
                    
        else:

            # TRAIN
            if model_name == "SumEmbeddings":
                SE_model_train(train_data)
            elif model_name == "AvgEmbeddings":
                AE_model_train(train_data)
            elif model_name == "BiGRU":
                BG_model_train(train_data)
            elif model_name == "BiLSTM":
                BL_model_train(train_data)
            elif model_name == "BERT":
                BERT_model_train(train_data)


            # EVALUATE
            if model_name == "SumEmbeddings":
                SE_model_test(test_data)
            elif model_name == "AvgEmbeddings":
                AE_model_test(test_data)
            elif model_name == "BiGRU":
                BG_model_test(test_data)
            elif model_name == "BiLSTM":
                BL_model_test(test_data)
            elif model_name == "BERT":
                BERT_model_test(test_data)
                
if argObj.wandb:
    wandb.agent(sweep_id, function = main, count = 6)
else:
    main()