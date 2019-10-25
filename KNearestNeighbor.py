import operator
from collections import Counter


class KNearestNeighbor:
    def __init__(self, k):
        self.k = k  # k denotes the number of clusters
        self.result = []  # result will return the actual value in prediction

    def fit(self, X_train, y_train):  # creates a function to fit X_train and y_train
        self.X_train = X_train
        self.y_train = y_train
        print("Training Done")

    def predict(self, X_test):  # creates a function to predict the value
        for j in X_test:
            distance = {}
            counter = 1  # initialize counter as 1
            for i in self.X_train:
                s = 0
                for t in range(len(j)):
                    s += (j[t] - i[t]) ** 2  # here s represents the sum of the squares
            distance[counter] = s ** 1 / 2  # it helps to calculate the distance between the nearest points
            counter += 1  # increment by 1
            distance = sorted(distance.items(), key=operator.itemgetter(1))  # sort the distance in ascending order
            self.result.append(self.classify(distance=distance[:self.k]))  # the required result stored in result
            # variable
        return self.result

    def classify(self, distance):  # now using classify function to collect the index of the outputs
        label = []

        for i in distance:
            label.append(self.y_train[i[0]])

        return Counter(label).most_common()[0][0]
