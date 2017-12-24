import numpy as np
from math import sqrt
#import matplotlib.pyplot as plt
import warnings
#from matplotlib import style
from collections import Counter
#style.use('fivethirtyeight')
import pandas as pd
import random

#dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
#k and r are classes, [1,2] etc are features.
#new_features = [5,7]

"""
[[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0],new_features[1],s=60)
plt.show()
"""

def k_nearest_neighbors(data,predict,k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])
    #note:i[1] is "group" not "euclidean_distance"!
    votes = [i[1] for i in sorted(distances)[:k]]  #sorting arrary"distances" in order till k
    #print(Counter(votes).most_common(2))
    #print(votes)
    vote_result = Counter(votes).most_common(1)[0][0] #counter gives result of [class, # of times it exists]
                                                      #[0][0] gives the class only
    confidence = Counter(votes).most_common(1)[0][1] / k
    #print(vote_result, confidence)
    return vote_result, confidence

#result = k_nearest_neighbors(dataset,new_features,k=5)
#print(result)

accuracies = []
for i in range(25):
    df = pd.read_csv("breast-cancer-wisconsin.data.txt")
    df.replace('?',-99999, inplace = True)
    df.drop(['id'], 1, inplace=True)
    #print(df.head())
    full_data = df.astype(float).values.tolist()     #convert data to float form
    random.shuffle(full_data)                        #shuffle(reorganize) data

    test_size = 0.2
    train_set = {2:[], 4:[]}     #Curly braces create dictionaries or sets, square
                                 #brackets create lists. here it creates a dictionary
                                 #with two classes: 2 and 4, and define each class
                                 #as a null(empty) array
    test_set = {2:[], 4:[]}
    train_data = full_data[:-int(test_size*len(full_data))]  #=[:-a] means equals from the first item
                                                             #to the last a+1 one. try A=[2,3,4] and A[:-1]
    test_data = full_data[-int(test_size*len(full_data)):]

    for i in train_data:
        train_set[i[-1]].append(i[:-1])    #train_set[i[-1]] define whether it belongs to 2 or 4 class.
                                           #append(i[:-1]) add each row of train_data into train_set except last column
    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    #Use group to define different classes, easier for "correct" judgement.
    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct += 1
            #else:
                #print(confidence)       #show up confidence of incorrect items. 
            total +=1
    #print('Accuracy', correct/total)
    accuracies.append(correct/total)
print(sum(accuracies)/len(accuracies))
