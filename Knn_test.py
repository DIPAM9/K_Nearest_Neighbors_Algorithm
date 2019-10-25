import numpy as np
import pandas as pd
from KNearestNeighbor import KNearestNeighbor

# take the data
data = pd.read_csv('Social_Network_Ads')
# replace the value of male and female using 0 and 1
data['Gender'] = data['Gender'].replace({'Male': 0, 'Female': 1})

# take X as input and y as output
X = data.iloc[:, 1:4].values
y = data.iloc[:, -1].values

# using the function train_test_split from sklearn library
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standarize the values using the function StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Now transform the values in standard form
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# using KNearestNeighbor class
knn = KNearestNeighbor(k=17)

# fitting in the algorithm
knn.fit(X_train, y_train)

# predict the value using predict() function
y_pred = knn.predict(np.array(X_test).reshape(len(X_test), len(X_test[0])))
# calculate the accuracy sore
from sklearn.metrics import accuracy_score

print("Accuracy: ", accuracy_score(y_test, y_pred))


# creating a function which will give us the output whether the person will purchase or not
def predict_new():
    age = (int(input("Enter the age: ")))
    salary = (int(input("Enter Salary: ")))
    gender = int(input("Enter your gender(for Male press 0, or for female press 1): "))
    X_new = np.array([[gender], [age], [salary]]).reshape(1, 3)
    X_new = scaler.transform(X_new)
    result = knn.predict(X_new)
    if result == 0:
        print("Will not purchase")
    else:
        print("Will purchase")


predict_new()  # call the function
