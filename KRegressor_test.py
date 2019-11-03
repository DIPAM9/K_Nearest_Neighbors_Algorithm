import numpy as np
import pandas as pd

from KNearestNeighbor import KNearestRegressor

data = pd.read_csv('Social_Network_Ads')
data['Gender'] = data['Gender'].replace({'Male': 0, 'Female': 1})
X = data.iloc[:, 2:4].values
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn = KNearestRegressor(k=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(np.array(X_test).reshape(len(X_test), 2))

from sklearn.metrics import r2_score

accuracy = r2_score(y_test,y_pred)

print(abs(accuracy))



