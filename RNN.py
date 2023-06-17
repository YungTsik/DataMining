import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

corona = pd.read_csv("Data/data.csv")

#Rename
corona.rename(columns={"Daily tests": "Tests"}, inplace=True)
#Get only Greece
covidGR = corona[corona["Entity"] == "Greece"]
#Get only before 01/01/2021
covidGR.sort_values(by=["Date"],inplace=True)
idx = covidGR.index.get_loc(covidGR[covidGR["Date"] == "2021-01-01"].index[0])
covidGR = covidGR.iloc[0:idx+1]

#Get only Tests and Cases
covidGR = covidGR[["Tests","Cases"]]
#Fill NaN values
covidGR.Cases  = corona.groupby("Entity").Cases.transform(lambda x: x.fillna(0))
covidGR.Tests  = corona.groupby("Entity").Tests.transform(lambda x: x.fillna(0))
#We want to have the daily cases for each day not all the cases till that day
#We will find the daily cases for each day, by substracting from the number of cases untill the specific date,
#the number of cases untill the previous day
covidGR["Daily Cases"] = np.NaN
for index in range(len(covidGR)):
    if index==0:
        covidGR["Daily Cases"].iloc[index] = covidGR["Cases"].iloc[index]
    else:
        covidGR["Daily Cases"].iloc[index] = covidGR["Cases"].iloc[index] - covidGR["Cases"].iloc[index-1]

#Make Cases/Tests
covidGR["DailyCases/Tests"] = covidGR["Daily Cases"]/covidGR["Tests"]

#Keep only
covidGR = covidGR["DailyCases/Tests"]
#Drop rows where inf cases/tests
# Replace infinite updated data with nan
covidGR.replace([np.inf, -np.inf], np.nan, inplace=True)
# Drop rows with NaN
covidGR.dropna(inplace=True)
#Change precision
covidGR = covidGR.round(3)


covidGR = covidGR.to_numpy()
x_train = [] #days
y_train = [] #days + 3
for i in range(2,covidGR.shape[0]-3):
    x_train.append([covidGR[i-2],covidGR[i-1],covidGR[i]])
    y_train.append(covidGR[i+3])


x_train = np.array(x_train)
y_train = np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(36, input_shape=(3,1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=60)


test_data = [covidGR[covidGR.shape[0]-3],covidGR[covidGR.shape[0]-2],covidGR[covidGR.shape[0]-1]]
test_data = np.array(test_data)
test_data = test_data.reshape((1, 3, 1))

print(test_data)
print(model.predict(test_data))

