import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn import utils
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score, mean_squared_error

pd.options.mode.chained_assignment = None  # default='warn'

corona = pd.read_csv("Data/alteredData.csv")
corona.drop(['Continent', 'Average temperature per year', 'Hospital beds per 1000 people', 'Medical doctors per 1000 people', 'GDP/Capita', 'Population', 'Median age', 'Population aged 65 and over (%)'], axis=1, inplace=True)
corona.dropna(inplace=True)

coronaGR = corona[corona['Entity'] == 'Greece']
coronaGR = coronaGR.loc[coronaGR['Date'] < '2021-01-02']
coronaGR['Pososto'] = np.nan

coronaGR['Pososto'].iloc[0] = float(coronaGR['Cases'].iloc[0]) / float(coronaGR['Tests'].iloc[0])

for i in range(1, len(coronaGR)):
    coronaGR['Pososto'].iloc[i] = float(coronaGR['Cases'].iloc[i] - coronaGR['Cases'].iloc[i-1]) / coronaGR['Tests'].iloc[i]

X = list()
Y = list()

coronaGR = coronaGR.round(2)
coronaPososto = coronaGR.loc[:, 'Pososto']

coronaGR.to_csv('Data/greece.csv')

for i in range(2,coronaGR.shape[0]-3):
    X.append([float(coronaGR['Pososto'].iloc[i-2]), float(coronaGR['Pososto'].iloc[i-1]), float(coronaGR['Pososto'].iloc[i])])
    Y.append([float(coronaGR['Pososto'].iloc[i+3])])


X = np.reshape(X, [-1,3])
Y = np.reshape(Y, [-1,1])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

lin = svm.SVR(kernel='linear', C=1000.0)
lin.fit(X_train, y_train.ravel())

poly = svm.SVR(kernel='poly', C=1000.0, degree=2)
poly.fit(X_train, y_train.ravel())

rbf = svm.SVR(kernel='rbf', C=1000.0, gamma=0.15)
rbf.fit(X_train, y_train.ravel())


# plt.figure(figsize=(16,8))
# plt.scatter(X, Y, color='red', label='Data')
# plt.plot(X,rbf.predict(X), color='green', label='RBF Model')
# plt.plot(X,poly.predict(X), color='orange', label='Polynomial Model')
# plt.plot(X,lin.predict(X), color='blue', label='Linear Model')
# plt.legend()
# plt.show()

print(rbf.predict(np.array([[0.03, 0.03, 0.02]])))

r2 = r2_score(y_test, rbf.predict(X_test))
print(f'R2 score: {r2}')

mse = mean_squared_error(y_test, rbf.predict(X_test))
print(f'MSE score: {mse}')