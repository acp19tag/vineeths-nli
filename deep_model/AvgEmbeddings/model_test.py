# Imports
import numpy as np
from tensorflow.keras.models import load_model
from deep_model.AvgEmbeddings.preprocess import preprocess_testdata
from deep_model.AvgEmbeddings.parameters import *


# Uncomment for generating plots. Requires some libraries (see below)
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from utils.plot_confusion_matrix import plot_confusion_matrix
from tensorflow.keras.utils import plot_model



# Tests the AvgEmbedding model using the data passed as argument
def AE_model_test(data, wandb):
    
    # overwrite the parameters
    BATCH_SIZE = wandb.config.batch_size

    # Preprocess the data
    test_data = preprocess_testdata(data)

    with open('./results/AvgEmbeddings/AvgEmbeddings.txt', 'w') as output_file:
        output_file_root = open('deep_model.txt', 'w')

        # Check if model exists at 'model' directory
        try:
            model = load_model('./model/AvgEmbeddings.h5')
        except:
            print("Trained model does not exist. Please train the model.\n")
            exit(0)

        # Evaluate the loaded model with test data
        loss, accuracy = model.evaluate(x=[test_data[0], test_data[1]], y=test_data[2], batch_size=BATCH_SIZE)
        print("Test Loss: {:.2f}, Test Accuracy: {:.2f}%\n".format(loss, (accuracy*100)))

        # Obtain the predicted classes
        Y_pred = model.predict([test_data[0], test_data[1]])
        Y_pred = np.argmax(Y_pred, axis=1)
        Y_test = np.argmax(test_data[2], axis=1)

        # Write output to file
        for ind in range(Y_pred.shape[0]):
            if Y_pred[ind] == 0:
                output_file.write("contradiction\n")
                output_file_root.write("contradiction\n")
            elif Y_pred[ind] == 1:
                output_file.write("neutral\n")
                output_file_root.write("neutral\n")
            elif Y_pred[ind] == 2:
                output_file.write("entailment\n")
                output_file_root.write("entailment\n")

    # # Uncomment for generating plots.
    # confusion_mtx = confusion_matrix(Y_test, Y_pred)
    # plot_confusion_matrix(confusion_mtx, "AvgEmbeddings", classes=range(2))

    target_names = [f"Class {i}" for i in range(CATEGORIES)]
    classification_rep = classification_report(Y_test, Y_pred, target_names=target_names, output_dict=True)
    wandb.log(classification_rep)

    # plt.figure()
    # sns.heatmap(pd.DataFrame(classification_rep).iloc[:-1, :].T, annot=True)
    # plt.savefig('./results/AvgEmbeddings/classification_report.png')
    # # plt.show()
    # plot_model(model, to_file='./results/AvgEmbeddings/model_plot.png', show_shapes=True, show_layer_names=True)
    