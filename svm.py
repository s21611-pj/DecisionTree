import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.plotting import plot_decision_regions
from sklearn import svm, model_selection, decomposition, metrics

"""
Authors:
- Badysiak Pawel - s21166
- Turek Wojciech - s21611
Implementation of Decision Tree and Support Vector Machine
There are two cases: 
    a) Banknote Authenticity - check if banknote is authentic
        Link to dataset( number 5 on the list): https://machinelearningmastery.com/standard-machine-learning-datasets/       
    b) Phoneme - check if phoneme is nasal (Class 0) or oral (Class 1)
        Link to dataset: https://github.com/jbrownlee/Datasets/blob/master/phoneme.csv
        Link to description: https://github.com/jbrownlee/Datasets/blob/master/phoneme.names
        
    Install before:
        pip install wheel
        pip install pandas
        pip install mlxtend
        pip install numpy
"""


def classify_svc(dataset, split_size, kernel, C, gamma, dataset_name):
    """
    This method is responsible for training SVC model with dataset and it's configuration
    At the end it classifies the result and compares it to the test part.
    """
    # Prints CSV's file headers
    print(dataset.head(5))

    # Split inputs into X and output into Y
    x, y = dataset.drop('class', axis=1), dataset['class']

    X_train, X_test, y_train, y_test = model_selection.train_test_split(x, np.array(y), test_size=split_size)

    # Assigning parameters for training model and evaluate train method
    svc = svm.SVC(kernel=kernel, C=C, gamma=gamma)
    svc.fit(X_train, y_train)

    # Print all results and compare them visually
    y_model_result = svc.predict(X_test)
    print(f"Sample results for {dataset_name}")
    print(f"Model: {y_model_result[0:20]}")
    print(f"Target: {y_test[0:20]}")

    # Decompose and transform data for showing it on X/Y chart
    pca = decomposition.PCA(n_components=2)
    X_train2 = pca.fit_transform(X_train)
    svc.fit(X_train2, y_train)
    plot_decision_regions(X_train2, y_train, clf=svc, legend=2)
    plt.title(dataset_name)
    plt.show()

    score = metrics.accuracy_score(y_test, y_model_result)
    return score


if __name__ == "__main__":
    # Load files
    banknote = pd.read_csv("data_banknote_authentication.csv")
    phoneme = pd.read_csv("phoneme.csv")

    # Evaluation of classification method with provided dataset and parameters
    # Is responsible for training and testing data
    banknote_svc_result = classify_svc(banknote, 0.2, 'rbf', 10, 0.00001, 'Banknote Authenticity')
    print(f"Precision of SVC model:\nBanknote dataset: {banknote_svc_result}")
    phoneme_svc_result = classify_svc(phoneme, 0.2, 'rbf', 10000, 0.1, 'phoneme')
    print(f"Precision of SVC model:\nphoneme dataset: {phoneme_svc_result}")
