import pandas as pd
import quandl, math, datetime
import numpy as np     #math library
from sklearn import preprocessing, cross_validation, svm     #preprocessing is used for scaling data from -1 to +1
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt    #for plot

from matplotlib import style       #for making the plot descent
import pickle                      #for pickling

style.use('ggplot')                #specify which descent plot you want to use

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)    #fill places where there is n/a 

forecast_out = int(math.ceil(0.01*len(df)))
#print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)  #shift(neg) - upward, shift(pos) - downward


#print(df)       #print(df.tail()) is last 5 rows of df

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)  #this step is time consuming
X_lately = X[-forecast_out:]  #remember there is a "-" before forecast_out
                              #X_lately needs to come before X
X = X[:-forecast_out]
df.dropna(inplace=True)   #delete all rows that contains NaN, !!because after
                          #the shift (upward), certain # of rows has NaN at 'label'
                          #this dropna helps to drop all those line away.
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

#clf = LinearRegression()   #clf for classification, using linear regression
clf = LinearRegression(n_jobs=-1)  #n_jobs define how many group of data in training set
                                   #are runing at the same time. default is 1.  -1 means
                                   #as many as possible
#clf = svm.SVR()             #classification using svm   #clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train)
#using pickle to store the trained date structure to save time
with open('linaerregression.pickle','wb') as f: #wb-write in binary mode
    pickle.dump(clf, f)   #takes a serializable Python data structure, serializes it into a binary,
                          #and saves it to an open file.

pickle_in = open('linaerregression.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test,y_test)
#print(accuracy)
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name         #iloc[-1] gives the last row in df, note that the df
                                     #does not include predict rows
last_unix = last_date.timestamp()    #timestamp() converts last_date to a proper format 
one_day = 86400                      #seconds in a day
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
                                     #set each column in rows without the last column (i.e.
                                     #the 'Forecast') as NaN, the set 'Forecast'=forecast_set.

#print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
