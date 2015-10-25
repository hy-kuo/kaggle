import pandas
import csv
from sklearn import svm
import numpy as np

titanic = pandas.read_csv("train.csv")
titanic_test = pandas.read_csv("test.csv")
#Replace all the missing values in the Age column of titanic.
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())

# Replace all the occurences of male with the number 0.
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0

# Replace all the occurences of female with the number 1.
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

# Replace all the missing values in the Embarked column with S
titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

#Assign the code 0 to S, 1 to C and 2 to Q. Replace each value in the Embarked column with its corresponding code.
titanic.loc[titanic["Embarked"]=="S", "Embarked"] = 0
titanic.loc[titanic["Embarked"]=="C", "Embarked"] = 1
titanic.loc[titanic["Embarked"]=="Q", "Embarked"] = 2
titanic_test.loc[titanic_test["Embarked"]=="S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"]=="C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"]=="Q", "Embarked"] = 2

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())


predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
predictors = ["Pclass", "Sex", "Age", "Parch", "Embarked"]
predictors = ["Pclass", "Sex", "Age", "Parch"]

X = titanic[predictors]
y = titanic["Survived"]
clf = svm.SVC()
clf.fit(X, y)
predictions = clf.predict(X)
accuracy = 1 - np.count_nonzero(predictions-y) / len(y)
prediction_file = open("mymodel2.csv", "wb")
prediction_file_object = csv.writer(prediction_file)
X_test = titanic_test[predictors]
predictions_test = clf.predict(X_test)
prediction_file_object.writerow(["PassengerId", "Survived"])
PassengerIds = titanic_test["PassengerId"].values
for i in xrange(len(PassengerIds)):
    prediction_file_object.writerow([PassengerIds[i],predictions_test[i]])
