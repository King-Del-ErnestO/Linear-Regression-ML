import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#%matplotlib inline

dataset = pd.read_csv("weather.csv")
#Finding out details about the data we are working with
print(dataset.shape)

print(dataset.describe())

#Determining the parts of the data we are using and finding a relationship between them
dataset.plot(x='MinTemp', y='MaxTemp', style='o')
plt.title('MinTemp vs MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel("MaxTemp")
plt.show()

#Looking at the average of our predictor variables
plt.figure(figsize=(15, 10))
plt.tight_layout()
seabornInstance.distplot(dataset['MaxTemp'])
plt.show()

#Storing the sets into variables and Slicing the data
x = dataset['MinTemp'].values.reshape(-1, 1)
y = dataset['MaxTemp'].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Training the algorithm with our Training dataset
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#To retrieve the intercept:
print('Intercept:', regressor.intercept_)
print('Coefficient/Slope', regressor.coef_)

#It's time to make predictions now
y_pred = regressor.predict(x_test)

#Comparing Predicted Values to Actual values
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

#
df1 = df.head(25)
df1.plot(kind='bar', figsize=(16, 10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

plt.scatter(x_test, y_test, color='gray')
plt.plot(x_test, y_pred, color='red', linewidth=2)
plt.show()

#Evaluating the performance of the algortihm

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print(regressor.score(y_test, y_pred))
#Well you can Train your model with alot more data, use parameter tuning, use a different predictor
#variable, to get better results with lesser error.
