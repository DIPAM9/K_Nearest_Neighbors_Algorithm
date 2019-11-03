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
            weight = {}
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

    def weight(self, X_test):  # created  a weight function for Weighted KNN algorithm
        distances = {}
        counter = 1  # initialize all the variables
        for i in self.X_train:
            distances[counter] = ((X_test[0][0] - i[0]) ** 2 + (X_test[0][1] - i[1]) ** 2) ** 1 / 2
            counter += 1
        distances = sorted(distances.items(), key=operator.itemgetter(1))
        print(distances)  # calculate the distances & printed it
        weights = []  # initialize the variable weight
        for i in distances:
            for j in range(len(i)):
                weights.append(1 / i[j])
        print(weights)  # now reversing the distance we have calculated the weights & printed it
        labels = []
        for i in distances[:self.k]:
            labels.append(self.y_train[i[0]])
        print(labels)  # now checking the labels whether it is purchased or not
        purchased = []
        not_purchased = []
        for i in range(len(labels)):
            if labels[i] == 1:  # here 1 denotes the purchased & 0 denotes not purchased
                purchased.append(weights[i])
            else:
                not_purchased.append(weights[i])
        p = 0
        n = 0
        for i in purchased:
            p = p + i  # summation of all weights to compare two points
        for i in not_purchased:
            n = n + i

        print(purchased)
        print(not_purchased)
        print(p)
        print(n)
        if p >= n:
            print("Will purchase ")
        else:
            print("Will not purchase")


class KNearestRegressor():
    def __init__(self, k):
        self.k = k
        self.result = []

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        print("Training done")

    def predict(self, X_test):  # creates a function to predict the value
        for j in X_test:
            distance = {}
            weight = {}
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


    def classify(self, distance):
        label = []
        for i in distance:
            label.append(self.y_train[i[0]])
        return sum(label)/len(label)

