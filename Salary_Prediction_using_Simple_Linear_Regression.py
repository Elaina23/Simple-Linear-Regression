import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score


# load datasets/import file
df = pd.read_csv(r'C:\Users\elena\Desktop\Salary_Data.csv') #read the csv file ('r' is put before the path string to address any special characters in the path, such as '\')
#print(df)
# to have a look at the dataset
df.head()

# summarize the data/descriptive exploration on the data
df.describe()

# plotting the features 
plott = df[['YearsExperience','Salary']]
plott.hist()
plt.show()

# scatter plot to see how linear the relation is
plt.scatter(plott.YearsExperience, plott.Salary,  color='cyan')
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

# splitting the dataset into train and test sets, 80% for training and 20% for testing
# create a mask to select random rows using np.random.rand() function
msk = np.random.rand(len(df)) < 0.8
train = plott[msk]
test = plott[~msk]

# Train data distribution
plt.scatter(train.YearsExperience, train.Salary,  color='red')
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

# modelling data using sklearn package. Sklearn would estimate the intercept and slope
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['YearsExperience']])
train_y = np.asanyarray(train[['Salary']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ', regr.intercept_)


# Plot outputs. Plot the fit line over the data
plt.scatter(train.YearsExperience, train.Salary,  color='red')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("YearsExperience")
plt.ylabel("Salary")


# Evaluate model accuracy by finding MSE
test_x = np.asanyarray(test[['YearsExperience']])
test_y = np.asanyarray(test[['Salary']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )

