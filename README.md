# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program
2. Import numpy as np 3.Plot the points
3. IntiLiaze thhe program
4. End the program

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: DHARSAN KUMAR R
RegisterNumber:212223240028
*/
```
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        #Calculate predictions
        predictions=(X).dot(theta).reshape(-1,1)
        
        #Calculate errors
        errors=(predictions-y).reshape(-1,1)
        #Update theto using gradient descent
        theta=learning=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
#Learn model Parameters
theta= linear_regression(X1_Scaled,Y1_Scaled)
#Predict data value for a new value point
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")

## Output:
![image](https://github.com/DHARSAN23014208/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/149365413/4b4acc2c-cd9f-4ae3-a022-51c6b29ed9fe)
data.head()
![image](https://github.com/DHARSAN23014208/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/149365413/fe08198c-40e1-457a-86a6-a02f889bae64)




![image](https://github.com/DHARSAN23014208/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/149365413/71f7417a-8e3d-4187-a4eb-a5bf9976a767)
![image](https://github.com/DHARSAN23014208/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/149365413/878a5df7-e042-4122-8a6f-a4b44a4e1433)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
