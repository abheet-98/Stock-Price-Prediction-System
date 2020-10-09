# Load libraries
import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Loading Dataset
dataset = pandas.read_csv('DS1.csv')

x=dataset.iloc[:,1].values
y=dataset.iloc[:,4].values

x=x.reshape(-1,1)
y=y.reshape(-1,1)

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

reg=LinearRegression()
reg.fit(x_train,y_train)

y_predict=reg.predict(x_test)
a=float(input("Enter the opening price "))
y_pred=reg.predict(a)

plt.scatter(x_train,y_train,color='blue',marker='o')
plt.plot(x_train,reg.predict(x_train),color='red')
plt.xlabel('Open Price')
plt.ylabel('Close Price')
plt.show()

print("Closing Price is ",y_pred);
