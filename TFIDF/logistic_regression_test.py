# Imports
import pickle
from TFIDF.TFIDF_features import TFIDF_features
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from utils.plot_confusion_matrix import plot_confusion_matrix
from tensorflow.keras.utils import plot_model
# Uncomment for generating plots. Requires some libraries (see below)
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Loads and tests the logistic regression model
def logistic_regression_test(test_data):
    # Obtain the TFIDF features
    test_feature, test_label = TFIDF_features(test_data, "test")

    # Loads the logistic regression model from pickle file
    with open('./model/LR.pickle', "rb") as file:
        LR_model = pickle.load(file)

    # Tests the logistic regression model
    pred_labels = LR_model.predict(test_feature)

    with open('./tfidf.txt', "w") as file:
        for item in pred_labels:
            if item == 0:
                file.write("contradiction\n")
            elif item == 1:
                file.write("neutral\n")
            elif item == 2:
                file.write("entailment\n")
            else:
                pass

    # Evaluate and print the results
    # score = LR_model.score(test_feature, test_label) * 100
    # print("The classification accuracy for Logistic regression with TF-IDF features is {:.2f}%.".format(score))

    # Uncomment for generating plots.
    confusion_mtx = confusion_matrix(test_label, pred_labels)
    plot_confusion_matrix(confusion_mtx, "Logistic Regression", classes=range(3))

    target_names = ["Class {}".format(i) for i in range(2)]
    classification_rep = classification_report(test_label, pred_labels, target_names=target_names, output_dict=True)

    plt.figure()
    sns.heatmap(pd.DataFrame(classification_rep).iloc[:-1, :].T, annot=True)
    plt.savefig('./results/Logistic Regression/classification_report.png')
    # plt.show()