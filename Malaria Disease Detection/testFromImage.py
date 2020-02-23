import pandas as pd
import pickle

model = pickle.load(open("malaria.pkl",'rb'));

dataframe = pd.read_csv("csv/dataset_testing1111.csv")

x = dataframe.drop(['Label'],axis=1)
y = dataframe['Label']
for i in range(len(x)):
    predictions = model.predict(x[i:i+1])
    print(predictions)