from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import graphviz
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    filepath = "./data/column_3C.dat"
    names = ["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle", "sacral_slope", "pelvic_radius",
             "degree_spondylolisthesis", "class"]

    df = pd.read_csv(filepath, sep=" ", header=None, names=names)

    x = df.drop('class', axis=1)
    y = df['class']

    train_data, test_data, train_label, test_label = train_test_split(x, y, test_size=0.2)

    train_list = []
    test_list = []
    clfs = []
    for i in range(1,20):
        clf = RandomForestClassifier(n_estimators=i)
        clf = clf.fit(train_data, train_label)

        train_prediction = clf.predict(train_data)
        train_accuracy = accuracy_score(train_label, train_prediction)

        test_prediction = clf.predict(test_data)
        clfs.append(clf)
        test_accuracy = accuracy_score(test_label, test_prediction)

        # print(str(i) + ': ' + str(train_accuracy) + str(test_accuracy))
        print(str(i) + ': ' + str(test_accuracy))

        # obj = [i, train_accuracy]
        # train_list.append(obj)

        obj = [i, test_accuracy]
        test_list.append(obj)

    # train_list = sorted(train_list, key=lambda x:x[1], reverse=True)
    test_list = sorted(test_list, key=lambda x:x[1], reverse=True)

    # print(train_list)
    # print(test_list)
    order = []
    for i in range(1, 20):
        order.append(test_list[i-1][0])
    print(order)

    clf_loc = test_list[0][0]
    # num_of_estimator = clf_loc + 1

    print('Best forest: ' + str(clf_loc) + ' - ' + str(test_list[0][1]))

    best_forest = clfs[clf_loc-1]

    print('Component Tree:')

    for i in range(0, clf_loc):
        dot_data = tree.export_graphviz(best_forest.estimators_[i], out_file=None, filled=True,
                                        feature_names=["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle",
                                                       "sacral_slope", "pelvic_radius", "degree_spondylolisthesis"],
                                        class_names=["Hernia", "Normal", "Spondylolisthesis"], rounded=True,
                                        special_characters=True)

        graph_training = graphviz.Source(dot_data)
        graph_training.render('Graph' + str(i+1), view=True)
        tree_test_prediction = best_forest.estimators_[i].predict(test_data)
        tmp_list = []
        for result in tree_test_prediction:
            if str(int(result)) == '0':
                tmp_list.append('DH')
            elif str(int(result)) == '1':
                tmp_list.append('NO')
            else:
                tmp_list.append('SL')

        print(str(i+1) + ': ' + str(accuracy_score(test_label, tmp_list)))
        print(confusion_matrix(test_label, tmp_list))


    importances = best_forest.feature_importances_

    std = np.std([tree.feature_importances_ for tree in best_forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")

    for f in range(train_data.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the impurity-based feature importances of the forest
    if True:
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(train_data.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(train_data.shape[1]), indices)
        plt.xlim([-1, train_data.shape[1]])
        plt.show()

    nb = GaussianNB()
    nb = nb.fit(train_data, train_label)

    train_nb_prediction = nb.predict(train_data)
    train_nb_accuracy = accuracy_score(train_label, train_nb_prediction)

    test_nb_prediction = nb.predict(test_data)
    test_nb_accuracy = accuracy_score(test_label, test_nb_prediction)

    print("GaussianNB - Train: " + str(train_nb_accuracy))
    print("GaussianNB - Test: " + str(test_nb_accuracy))