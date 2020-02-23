import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pickle


dataframe = pd.read_csv("csv/dataset_553.csv")


x = dataframe.drop(['Label'],axis=1)
y = dataframe['Label']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=40)



model = RandomForestClassifier(max_depth=5, random_state=100)
model.fit(x_train,y_train)
pickle.dump(model,open("malaria.pkl",'wb'))


predictions = model.predict(x_test)
print(model.score(x_test,y_test))
