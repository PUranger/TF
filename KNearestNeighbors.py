#To get rid of the deprecationWarning for py3, replace cross_validation by model_selection
import numpy as np
#from sklearn import preprocessing, cross_validation, neighbors
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd

accuracies = []
for i in range(25):
    df = pd.read_csv('breast-cancer-wisconsin.data.txt')
    df.replace('?', -99999, inplace=True)    #replace all ? mark to -99999
    df.drop(['id'],1,inplace=True)           #dump the whole column of 'id'

    X = np.array(df.drop(['class'],1))
    y = np.array(df['class'])

    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size = 0.2)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size = 0.2)
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    #print(accuracy)

    #example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])
    #example_measures = example_measures.reshape(len(example_measures),-1)


    #prediction = clf.predict(example_measures)
    #print(prediction)

    accuracies.append(accuracy)

print(sum(accuracies)/len(accuracies))
