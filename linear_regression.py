
from sklearn import datasets,linear_model
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#Extracting Data!!!
features=datasets.load_diabetes()
df_features=pd.DataFrame(features.data[:,2:3])
df_target=pd.DataFrame(features.target)
X=np.array(df_features)
y=np.array(df_target)
X_train =  X[:-50]
y_train= y[:-50]
X_test = X[-50:]
y_test = y[-50:]


#Linear Regression implementation from scratch!!!!
#Using Gradient_Discent

no_iterations=2000
learning_rate=0.2
n=len(X_train)
m=0
b=0

def gradient(X_train,y_train,m,b,n):
  m_gradient = 0
  b_gradient = 0
  for i in range(n):
     y_current=y_train[i]
     x_current=X_train[i]
     m_gradient+=-2*(y_current-(m*x_current+b))*x_current/n
     b_gradient+=-2*(y_current-(m*x_current+b))/n
  return m_gradient,b_gradient

def new_values(m,b,X_train,y_train,learning_rate,no_iterations):
    for i in range(no_iterations):
        m_gradient,b_gradient=gradient(X_train,y_train,m,b,n)
        b=b-learning_rate*b_gradient
        m=m-learning_rate*m_gradient
    return b,m

b_latest,m_latest=new_values(m,b,X_train,y_train,learning_rate,no_iterations)
print(m_latest,b_latest)
y_predi=[]
for i in range(len(X_test)):
    y_predi.append(m_latest*X_test[i]+b_latest)


#SKLEARN implementation!!!!

regr=linear_model.LinearRegression()
regr.fit(X_train,y_train)
ans=(regr.predict((X_test)))


print("Predicted value by linear gradient__")
print(y_predi)
print("predicted value by SKLEARN implementation")
print(ans)
print("mean_squared_error by  linear gradient__")
print(mean_squared_error(y_test,y_predi))
print("mean_squared_error by SKLEARN implementation ")
print(mean_squared_error((ans),y_test))

#Plotting graph!!!

plt.scatter(X_test,y_test,color='red')
plt.xlabel('age')
plt.ylabel("sugar level")
plt.title('Linear regression')
plt.plot(X_test,ans,color='blue',label='SKLEARN')
plt.plot(X_test,y_predi,color='black',label="Gradient discent")
plt.legend()
plt.show()

