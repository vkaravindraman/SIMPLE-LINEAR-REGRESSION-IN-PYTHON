# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset

df = pd.read_csv('Salary_Data.csv')

x = df.iloc[:,:-1].values

y = df.iloc[:,1].values

# splitting the dataset into training set and test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3,
                                                    random_state = 0)

#fitting simple linear regression to Training set
from sklearn.linear_model import LinearRegression 
simplelinearRegression = LinearRegression()
simplelinearRegression.fit(x_train,y_train)

#Predicting the text result
y_predict = simplelinearRegression.predict(x_test)


#y_predict_val = simplelinearRegression.predict([[11.5]])

#implement graph (train set)

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, simplelinearRegression.predict(x_train), color = 'black')
plt.title('Salary vs Experience(train set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#implement graph (test set)
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, simplelinearRegression.predict(x_test), color = 'blue')
plt.title('Salary vs Experience(test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



