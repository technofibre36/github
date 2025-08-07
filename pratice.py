import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split

housing = pd.read_csv('BostonHousing.csv')

x = housing['indus','CHAS']
y = housing['medv']

x_train , x_test , y_train , y_test = train_test_split(x,y,random_state=104,test_size=0.25,shuffle=True)
print(housing)
print("x train data point")
print(x_train)
print("x test data point")
print(x_test)
print("y train data point")
print(y_train)
print("y test data point")
print(y_test)