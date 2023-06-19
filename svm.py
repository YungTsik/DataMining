import pandas as pd
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
pd.options.mode.chained_assignment = None  # default='warn'

#Reading the altered data from the CSV file and dropping unnecessary
corona = pd.read_csv("Data/alteredData.csv")
corona.drop(['Continent', 'Average temperature per year', 'Hospital beds per 1000 people', 'Medical doctors per 1000 people', 'GDP/Capita', 'Population', 'Median age', 'Population aged 65 and over (%)'], axis=1, inplace=True)
corona.dropna(inplace=True)

#Getting the data for Greece until 01-01-2021
coronaGR = corona[corona['Entity'] == 'Greece']
coronaGR = coronaGR.loc[coronaGR['Date'] < '2021-01-02']
coronaGR['Pososto'] = np.nan

#Filling the 'Pososto' column with the positivity rate
coronaGR['Pososto'].iloc[0] = float(coronaGR['Cases'].iloc[0]) / float(coronaGR['Tests'].iloc[0])

for i in range(1, len(coronaGR)):
    coronaGR['Pososto'].iloc[i] = float(coronaGR['Cases'].iloc[i] - coronaGR['Cases'].iloc[i-1]) / coronaGR['Tests'].iloc[i]

X = list()
Y = list()

#Rounding the data
coronaGR = coronaGR.round(2)
coronaPososto = coronaGR.loc[:, 'Pososto']

#Getting data for the target day (Y) and its three previous days (X)
for i in range(2,coronaGR.shape[0]-3):
    X.append([float(coronaGR['Pososto'].iloc[i-2]), float(coronaGR['Pososto'].iloc[i-1]), float(coronaGR['Pososto'].iloc[i])])
    Y.append([float(coronaGR['Pososto'].iloc[i+3])])

#Reshaping the lists
X = np.reshape(X, [-1,3])
Y = np.reshape(Y, [-1,1])

#Splitting the data into training and test groups
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#Training the model with the train data and with different kernels each time
lin = svm.SVR(kernel='linear', C=1000.0)
lin.fit(X_train, y_train.ravel())

poly = svm.SVR(kernel='poly', C=1000.0, degree=2)
poly.fit(X_train, y_train.ravel())

rbf = svm.SVR(kernel='rbf', C=1000.0, gamma=0.15)
rbf.fit(X_train, y_train.ravel())

#Our data for the prediction
predict_data = np.array([[0.03, 0.03, 0.02]])

#Calculating the R2 Score for the different kernels
rbf_r2 = r2_score(y_test, rbf.predict(X_test))
lin_r2 = r2_score(y_test, lin.predict(X_test))
poly_r2 = r2_score(y_test, poly.predict(X_test))

#Calculating the MSE Score for the different kernels
rbf_mse = mean_squared_error(y_test, rbf.predict(X_test))
lin_mse = mean_squared_error(y_test, lin.predict(X_test))
poly_mse = mean_squared_error(y_test, poly.predict(X_test))

#Printing the predictions and the scores for each kernel

print('RBF:')
print(f'Prediction: {rbf.predict(predict_data)}\nR2 Score: {rbf_r2}\nMSE Score: {rbf_mse}\n')

print('Linear:')
print(f'Prediction: {lin.predict(predict_data)}\nR2 Score: {lin_r2}\nMSE Score: {lin_mse}\n')

print('Polynomial:')
print(f'Prediction: {poly.predict(predict_data)}\nR2 Score: {poly_r2}\nMSE Score: {poly_mse}')