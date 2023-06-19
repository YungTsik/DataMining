import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


corona = pd.read_csv('Data/data.csv')
corona.dropna(inplace=True)

data = corona[corona['Entity'] == 'Greece']
data['Date'] = pd.to_datetime(data['Date'])
data = data.loc[data['Date'] < '2021-01-05']
data['Pososto'] = data['Cases'] / data['Daily tests']
data.to_csv("Data/greece2.csv",index=False)
# Extract the features (X) and target variable (y)
X = data[['Date']]
y = data['Pososto']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the SVM regressor
svr = SVR(kernel='rbf')

# Train the model
svr.fit(X_train, y_train)

y_pred = svr.predict(X_test)

mse = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)

# Get the last date from the dataset
last_date = data['Date'].max()

# Generate new dates for the next 3 days
next_three_days = pd.date_range(last_date + pd.DateOffset(days=1), periods=3)

# Create a new DataFrame with the new dates
X_pred = pd.DataFrame({'Date': next_three_days})

# Predict the 'Pososto' values for the next 3 days
y_pred_next_three_days = svr.predict(X_pred)
print(y_pred_next_three_days)