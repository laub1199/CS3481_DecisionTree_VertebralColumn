import pandas as pd
from sklearn import tree
from sklearn import preprocessing
import graphviz

def decisionTree():
    filepath = "./data/column_3C.dat"
    names = ["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle", "sacral_slope", "pelvic_radius",
             "degree_spondylolisthesis", "class"]

    df = pd.read_csv(filepath, sep=" ", header=None, names=names)
    df = df.sample(frac=1)

    numOfRow = len(df.index)

    trainingIndex = numOfRow * 0.8
    trainingIndex = int(trainingIndex)

    training_data = df.iloc[0:trainingIndex, 0:6]
    training_class = df.iloc[0:trainingIndex, -1]

    le = preprocessing.LabelEncoder()
    label = le.fit_transform(training_class)

    # clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(training_data, label)

    dot_data = tree.export_graphviz(clf, out_file=None, filled=True,
                                    feature_names=["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle",
                                                   "sacral_slope", "pelvic_radius", "degree_spondylolisthesis"],
                                    class_names=["Hernia", "Spondylolisthesis", "Normal"], rounded=True,
                                    special_characters=True)

    graph_training = graphviz.Source(dot_data)
    # graph_training.render('Graph', view=True)

    testing_data = df.iloc[trainingIndex:numOfRow + 1, 0:6]
    testing_class = df.iloc[trainingIndex:numOfRow + 1, -1]

    clf = clf.predict(testing_data)

    numberOfTestcase = numOfRow - trainingIndex
    numberOfMathes = 0
    for i in range(0, numberOfTestcase):
        clfToName = ""
        if (str(clf[i]) == "0"):
            clfToName = "DH"
        elif (str(clf[i]) == "1"):
            clfToName = "NO"
        else:
            clfToName = "SL"
        if (testing_class.iloc[i] == clfToName):
            numberOfMathes += 1

    print(str(numberOfMathes) + "/" + str(numberOfTestcase))

if __name__ == '__main__':
    decisionTree()
