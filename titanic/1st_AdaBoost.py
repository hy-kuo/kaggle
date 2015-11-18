import pandas as pd
import numpy as np
import pylab
import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
import re
import operator

TRAIN_PATH = "input/train.csv"
TEST_PATH = "input/test.csv"
SUBMISSION_PATH = "submission/submission_1st.csv"


def process_dataset(ds):
    # Deal with missing data: (1) kick (2) filled with median
    ds["Age"] = ds["Age"].fillna(ds["Age"].median())
    ds["Fare"] = ds["Fare"].fillna(ds["Fare"].median())
    ds["Embarked"] = ds["Embarked"].fillna('S')
    # Categorized
    ds.loc[ds["Sex"] == "male", "Sex"] = 0
    ds.loc[ds["Sex"] == "female", "Sex"] = 1
    ds.loc[ds["Embarked"] == 'S', "Embarked"] = 0
    ds.loc[ds["Embarked"] == 'C', "Embarked"] = 1
    ds.loc[ds["Embarked"] == 'Q', "Embarked"] = 2
    # Binning
    binning(ds, "Fare")
    binning(ds, "Age")
    # Create new feature
    ds["FamilySize"] = ds["SibSp"] + ds["Parch"]
    ds["NameLength"] = ds["Name"].apply(lambda x: len(x))
    titles = ds["Name"].apply(get_title)
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5,
                     "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8,
                     "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10,
                     "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
    for k, v in title_mapping.items():
        titles[titles == k] = v
    ds["Title"] = titles
    family_ids = ds.apply(get_family_id, axis=1)
    family_ids[ds["FamilySize"] < 3] = -1
    print(pd.value_counts(family_ids))
    ds["FamilyId"] = family_ids
    return ds


def visualize_tree(tree, features):
    with open("tree.dot", 'w') as f:
        export_graphviz(tree, out_file=f, feature_names=features)
    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")


def show_feature_lable(ds, f1, f2, l):
    ax = Axes3D(pylab.figure())
    ax.scatter(ds[f1], ds[f2], ds[l], color='r')
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_zlabel(l)
    plt.show()


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""


def binning(ds, name):
    instances = len(ds.axes[0])
    bin_size = int(np.sqrt(instances))
    nbin = int((instances+2)/bin_size)
    tmp = sorted(ds[name])
    cut = [tmp[i*bin_size] for i in range(nbin)]
    cut.append(tmp[instances-1])
    means = [(sum(tmp[i*bin_size:(i+1)*bin_size])/bin_size)
             for i in range(nbin)]
    for i in range(nbin):
        ds.loc[(cut[i] < ds[name]) & (ds[name] < cut[i+1]), name] = means[i]


def get_family_id(row):
    family_id_mapping = {}
    last_name = row["Name"].split(",")[0]
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            current_id = max(family_id_mapping.items(),
                             key=operator.itemgetter(1))[1]+1
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]


def read_dataset(path):
    ds = pd.read_csv(path)
    ds = process_dataset(ds)
    return ds


titanic = read_dataset(TRAIN_PATH)
# Select Feature: univariant selection
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
              "FamilySize", "Title", "FamilyId"]
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])
scores = -np.log10(selector.pvalues_)
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
#plt.show()

# Adaboosting
predictors = ["Pclass", "Sex", "Fare", "Title"]
alg = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=200, learning_rate=1.5, algorithm='SAMME.R', random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
# alg.fit(titanic[predictors], titanic["Survived"])
# print(alg.estimator_errors_)
print("[AdaBoostClassifier] {0:.2f}%".format(100*scores.mean()))

# Generate Submission
titanic_test = read_dataset(TEST_PATH)
alg.fit(titanic[predictors], titanic["Survived"])
predictions = alg.predict(titanic_test[predictors])
submission = pd.DataFrame({"PassengerId": titanic_test["PassengerId"], "Survived": predictions})
submission.to_csv(SUBMISSION_PATH, index=False)
