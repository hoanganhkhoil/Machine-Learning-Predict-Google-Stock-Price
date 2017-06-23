# Author: Khoi Hoang
# This algorithm is using data from Quandl to predict the stock price of Google Inc.
# in the next 30 days.

import pandas as pd
import quandl, math, datetime, time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from matplotlib import style

style.use('ggplot')



# Get dataset from Quandl ( 3223 x 12 ) - AT THIS TIME
data = quandl.get('WIKI/GOOGL')

# Get High - Low price percent change
data['HL_PCT'] = (data['Adj. High'] - data['Adj. Low']) / data['Adj. Low'] * 100

# Get Close - Open price percent change
data['PCT_change'] = (data['Adj. Close'] - data['Adj. Open']) / data['Adj. Open'] * 100

# Minimize number of feature from 12 to 4 (Adj. Close, HL_PCT, PCT_change, Adj. Volume)
data = data[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# Name the forecase column
forecast_col = 'Adj. Close'

# Days in future you want to predict the price ( ~33 days)
Days_in_future = int(math.ceil(0.01*len(data)))

# Optional- Fill Not Available data with -9999
data.fillna(-9999, inplace = True)
data.dropna(inplace = True)

# Full column of price - (optional)
# y_full = data[forecast_col]

# Price in future (shift the price column in 33 days ahead in the dataset)
# so the last 33 days in the dataset will not have a label.
data['Price'] = data[forecast_col].shift(-Days_in_future)


# Classify X and y
X = np.array(data.drop(['Price'],1))
y = np.array(data['Price'])

# Feature Scaling
X = preprocessing.scale(X)

# Get data with no label to actually predict the label
X_predict = X[-Days_in_future:]

# Get data up to the predictable date (with label) to train the algorithm
X = X[:-Days_in_future]
y = y[:-Days_in_future]


# Create training data vs testing data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.2)

# Create classifier - using Linear Regression
clf = LinearRegression()

#clf = svm.SVR() - default Kernel
#clf = svm.SVR(kernel='poly') - using Polynomial Kernel

# Training data
clf.fit(X_train, y_train)

# Save classifier into pickle so we will not need to re-train
# the classifier everytime we run the code.
# the traning process should be turned off next time running the code.
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf, f)

pickle_in = open('linearregression.pickle', 'rb')

clf = pickle.load(pickle_in)

# Testing data
accuracy = clf.score(X_test, y_test)

# Predict data
predictions = clf.predict(X_predict)

print predictions

data['Forecast'] = np.nan

last_date = data.iloc[-1].name
last_unix = (last_date - datetime.datetime(1970,1,1)).total_seconds() # Convert date to second
one_day = 86400
next_unix = last_unix + one_day


for i in predictions:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    data.loc[next_date] = [np.nan for _ in range(len(data.columns)-1)] + [i]


print accuracy

data['Adj. Close'].plot()
data['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
