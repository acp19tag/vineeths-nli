# Imports
import pickle
from sklearn.linear_model import LogisticRegression
from TFIDF.TFIDF_features import TFIDF_features


# Trains and stores a logistic regression model
def logistic_regression_train(train_data):
    # Obtain the TFIDF features
    
    # print(f'len(train_data) = {len(train_data)}') # DEBUG
    # if len(train_data) == 3:
    #     for idx, train_data_segment in enumerate(train_data):
    #         print(f'segment {idx} has len {len(train_data_segment)} and is type {type(train_data_segment)}')
    #         print(f'and the first element has len: {len(train_data_segment[0][0])}')
    #         print()
    #     print()
    
    # for idx, segment in enumerate(train_data):
    #     if idx < 2:
    #         train_data[idx] = segment[0] # attempted solution
    
    train_feature, train_label = TFIDF_features(train_data, "train")

    # Train the logistic regression model
    LR_model = LogisticRegression(random_state=0, max_iter=1000, solver='lbfgs', multi_class='auto')
    LR_model.fit(train_feature, train_label)

    # Save the logistic regression model as a pickle file
    with open('./model/LR.pickle', "wb") as file:
        pickle.dump(LR_model, file)

    print("Training complete.\n")