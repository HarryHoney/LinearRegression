from sklearn import datasets,linear_model
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
features=datasets.load_diabetes()
df_features=pd.DataFrame(features.data[:,2:3])
df_target=pd.DataFrame(features.target)
X=np.array(df_features)
y=np.array(df_target)
#print(X,y)

X_train =  X[:-50]
y_train= y[:-50]
X_test = X[-50:]
y_test = y[-50:]
'''
print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))
'''
regr=linear_model.LinearRegression()
regr.fit(X_train,y_train)
ans=(regr.predict((X_test)))
print(mean_squared_error((ans),y_test))
print(ans)
plt.scatter(X_test,y_test,color='red')
plt.xlabel('age')
plt.ylabel("sugar level")
plt.title('Linear regression')
plt.plot(X_test,ans,color='black')
plt.show()
print(regr.coef_,regr.intercept_)
