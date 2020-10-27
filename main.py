import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import pickle
import pandas as pd


def run():
    should_run = True
    while should_run:
        print("Menu:")
        print("1. Train a model")
        print("2. Make a prediction")
        selection = input("Please type 1 or 2 to make a selection or anything else to quit.\n")
        if selection == "1":
            train()
        elif selection == "2":
            predict()
        else:
            should_run = False
    print("Thank you for using Breast Cancer AI!")


def welcome():
    print("\n\n\n")
    print("Welcome to Breast Cancer AI!")
    print("This program can predict it a tumor is malignant or benign based on")
    print("inputs such as the mean radius and texture.\n")


def train():
    print("Training...")
    cancer = datasets.load_breast_cancer()
    pd.DataFrame(data=cancer['data'], columns=cancer['feature_names']).to_csv("data.csv", sep=",", index=False)

    print("\nFeature names:")
    print(cancer.feature_names)
    print("Target names:")
    print(cancer.target_names, "\n")

    x = cancer.data
    y = cancer.target

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

    clf = svm.SVC(kernel="linear")
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    acc = metrics.accuracy_score(y_test, y_pred)
    with open("model.pickle", "wb") as f:
        pickle.dump(clf, f)
    print("Done!")
    print("Accuracy: ", acc * 100, "%")


def predict():
    clf = None
    try:
        clf = pickle.load(open("model.pickle", "rb"))
    except FileNotFoundError:
        print("No model found.")
        return

    data = None
    try:
        path = input("Please enter the name of your input file (csv).\n")
        data = pd.read_csv(path)
    except FileNotFoundError:
        print("Invalid input.")
        return

    print(data)


if __name__ == '__main__':
    welcome()
    run()
