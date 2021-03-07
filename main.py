import pandas as pd
from sklearn import tree
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import graphviz
import matplotlib.pyplot as plt

def decisionTree(max_depth, outputTree, target_depth):
    filepath = "./data/column_3C.dat"
    names = ["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle", "sacral_slope", "pelvic_radius",
             "degree_spondylolisthesis", "class"]

    df = pd.read_csv(filepath, sep=" ", header=None, names=names)
    df = df.sample(n=len(df), random_state=42).reset_index(drop=True)

    numOfRow = len(df.index)

    trainingIndex = numOfRow * 0.8
    trainingIndex = int(trainingIndex)

    training_data = df.iloc[0:trainingIndex, 0:6]
    training_class = df.iloc[0:trainingIndex, -1]
    training_class_index= []
    for i in range(0, trainingIndex):
        if training_class.iloc[i] == "DH":
            training_class_index.append(0)
        if training_class.iloc[i] == "NO":
            training_class_index.append(1)
        if training_class.iloc[i] == "SL":
            training_class_index.append(2)

    le = preprocessing.LabelEncoder()
    label = le.fit_transform(training_class)

    # clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = tree.DecisionTreeClassifier(max_depth=max_depth, criterion='entropy')
    clf = clf.fit(training_data, label)

    dot_data = tree.export_graphviz(clf, out_file=None, filled=True,
                                    feature_names=["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle",
                                                   "sacral_slope", "pelvic_radius", "degree_spondylolisthesis"],
                                    class_names=["Hernia", "Spondylolisthesis", "Normal"], rounded=True,
                                    special_characters=True)

    graph_training = graphviz.Source(dot_data)
    if (outputTree):
        graph_training.render('Graph'+str(max_depth), view=True)

    clf_train = clf.predict(training_data)
    if (max_depth == target_depth):
        print(1-accuracy_score(training_class_index, clf_train))
        print(confusion_matrix(training_class_index, clf_train))

    numberOfTrainingMathes = 0

    for i in range(0, trainingIndex):
        clfToName = ""
        if (str(clf_train[i]) == "0"):
            clfToName = "DH"
        elif (str(clf_train[i]) == "1"):
            clfToName = "NO"
        else:
            clfToName = "SL"
        if (training_class.iloc[i] == clfToName):
            numberOfTrainingMathes += 1

    testing_data = df.iloc[trainingIndex:numOfRow + 1, 0:6]
    testing_class = df.iloc[trainingIndex:numOfRow + 1, -1]
    testing_class_index= []
    for i in range(0, numOfRow - trainingIndex):
        if testing_class.iloc[i] == "DH":
            testing_class_index.append(0)
        if testing_class.iloc[i] == "NO":
            testing_class_index.append(1)
        if testing_class.iloc[i] == "SL":
            testing_class_index.append(2)

    clf_test = clf.predict(testing_data)
    if (max_depth == target_depth):
        print(1-accuracy_score(testing_class_index, clf_test))
        print(confusion_matrix(testing_class_index, clf_test))

    numberOfTestcase = numOfRow - trainingIndex
    numberOfMathes = 0
    for i in range(0, numberOfTestcase):
        clfToName = ""
        if (str(clf_test[i]) == "0"):
            clfToName = "DH"
        elif (str(clf_test[i]) == "1"):
            clfToName = "NO"
        else:
            clfToName = "SL"
        if (testing_class.iloc[i] == clfToName):
            numberOfMathes += 1

    print(str(numberOfMathes) + "/" + str(numberOfTestcase))
    return 1 - (numberOfTrainingMathes / trainingIndex), 1 - (numberOfMathes / numberOfTestcase)


if __name__ == '__main__':
    train_list = []
    test_list = []
    levels = []
    for max_depth in range(1, 14):
        training_error, test_error = decisionTree(max_depth, False, 4)
        train_list.append(training_error)
        test_list.append(test_error)
        levels.append(max_depth)
    print(train_list)
    print(test_list)
    plt.plot(levels, train_list, label="Training Error")
    plt.plot(levels, test_list, label="Testing Error")
    plt.title("Training and Test Error Rate")
    plt.xlabel("Size of Tree")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.show()
