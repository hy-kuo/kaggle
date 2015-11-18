import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import KernelPCA
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.linear_model import Perceptron
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import re
import operator

TRAIN_PATH = "input/train.csv"
TEST_PATH = "input/test.csv"
SUBMISSION_PATH = "submission/submission_2nd.csv"


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
    ds["FamilyId"] = family_ids
    return ds


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


def pick_by_prob(df, prob):
    return np.sum(np.random.rand() >= np.cumsum(prob))


def adaboost_train(algorithms, df, target):
    N, aN = len(df), len(algorithms)
    prob = [1.0/N for x in range(N)]
    beta, algs = [], []
    cur = 0
    while(True):
        cur = (cur+1) % aN
        alg, pred = algorithms[cur]
        # draw N instances by prob
        X = [pick_by_prob(df, prob) for x in range(N)]
        alg.fit(df[pred].iloc[X, :], target.iloc[X])
        predicts = alg.predict(df[pred])
        correct = []
        for i, x in zip(range(N), df.index.get_values()):
            if target[x] == predicts[i]:
                correct.append(i)
        err_rate = (float)(N-len(correct))/N
        if err_rate > .5:
            return (beta, algs)
        bta = err_rate/(1-err_rate)
        for x in correct:
            prob[x] = prob[x]*bta
        norm = sum(prob)
        prob = [x/norm for x in prob]
        beta.append(bta)
        algs.append(algorithms[cur])


def adaboost_predict(beta, algs, df):
    N, aN = len(df), len(algs)
    preds = [0 for x in range(N)]
    norm = 0
    for i in range(aN):
        alg, pred = algs[i]
        tpred = alg.predict(df[pred])
        preds = [(p+tplog*np.log(1/beta[i])) for p, tplog in zip(preds, tpred)]
        norm += np.log(1/beta[i])
    th = norm/2
    preds = [(1 if x > th else 0) for x in preds]
    return preds


def pairwise_plot(features, target):
    fN = features.shape[1]
    positive = target.loc[target == 1].index
    negative = target.loc[target == 0].index
    for i in range(fN):
        for j in range(i+1, fN):
            plt.scatter(features.ix[positive, i], features.ix[positive, j], c='r')
            plt.hold(True)
            plt.scatter(features.ix[negative, i], features.ix[negative, j], c='g')
            plt.xlabel(features.columns[i])
            plt.ylabel(features.columns[j])
            plt.savefig(filename='analysis/'+features.columns[i]+'_'+features.columns[j]+'.png', format='png')
            plt.hold(False)


titanic = read_dataset(TRAIN_PATH)
# Select Feature: univariant selection
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
              "FamilySize", "Title", "FamilyId"]
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])
scores = -np.log10(selector.pvalues_)
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')

# RandomForestClassifier
# predictors = ["Pclass", "Sex", "Fare", "Title"]
# alg = RandomForestClassifier(random_state=1, n_estimators=150,
#                              min_samples_split=8, min_samples_leaf=4)
# scores = cross_validation.cross_val_score(alg, titanic[predictors],
#                                           titanic["Survived"], cv=3)
# print("[Random Forest select-4-features] {0:.2f}%".format(100*scores.mean()))

# Ensembleing: KNN + GradientBoostingClassifier
algorithms = [
    [GradientBoostingClassifier(random_state=1,
                                n_estimators=25, max_depth=3),
     ["Pclass", "Sex", "Age", "Fare",
      "Embarked", "FamilySize", "Title", "FamilyId"]],
    [KNeighborsClassifier(n_neighbors=4),
     ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked", "FamilyId"]]
]
kf = KFold(titanic.shape[0], n_folds=10, random_state=1)
predictions = []
for train, test in kf:
    train_target = titanic["Survived"].iloc[train]
    full_test_predictors = []
    for alg, predictors in algorithms:
        alg.fit(titanic[predictors].iloc[train, :], train_target)
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test, :].astype(float))[:, 1]
        full_test_predictors.append(test_predictions)
    test_predictions = (full_test_predictors[0] + full_test_predictors[1]) / 2
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)
predictions = np.concatenate(predictions, axis=0)
accuracy = sum([prd == sv for (prd, sv) in zip(predictions, titanic["Survived"])]) / float(len(predictions))
print("[KNN+GradientBoostingTree] {0:.2f}%".format(100*accuracy))


# Adaboost
algorithms = [[Perceptron(), ["Pclass", "Sex", "Fare"]], [RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=2, min_samples_leaf=2), ["Pclass", "Fare", "Sex", "Title", "FamilySize"]], [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Title", "FamilySize"]], [LinearSVC(), ["Pclass", "Sex", "Fare"]], [KNeighborsClassifier(n_neighbors=3), ["Sex", "Parch", "FamilySize", "FamilyId", "Embarked"]]]

kf = KFold(titanic.shape[0], n_folds=10, random_state=1)
predictions = []
for train, test in kf:
    beta, algs = adaboost_train(algorithms, titanic.iloc[train, :], titanic["Survived"].iloc[train])
    predictions.append(adaboost_predict(beta, algs, titanic.iloc[test, :]))
predictions = np.concatenate(predictions, axis=0)
acc = sum([prd == sv for (prd, sv) in zip(predictions, titanic["Survived"])]) / float(len(predictions))
print("[Adaboost: Perceptron-RF-GB-LinearSVM-KNN] {0:.2f}%".format(100*acc))

titanic_test = read_dataset(TEST_PATH)
predictions = adaboost_predict(beta, algs, titanic_test)
submission = pd.DataFrame({"PassengerId": titanic_test["PassengerId"], "Survived": predictions})
submission.to_csv(SUBMISSION_PATH, index=False)


# KernelPCA
titanic_test = read_dataset(TEST_PATH)
p = ["Pclass", "Sex", "Age", "Fare", "FamilySize"]
#pairwise_plot(titanic[p], titanic["Survived"])
kpca = KernelPCA(kernel="poly", tol=1e-3, gamma=100)
T_kpca = kpca.fit(titanic_test[p])
X_kpca = kpca.transform(titanic[p])
print("Found {0} columns".format(len(X_kpca[0])))
N = len(X_kpca)
fN = 10
X_kpca = pd.DataFrame({'kpca'+str(j+1): pd.Series(stats.zscore([X_kpca[i][j] for i in range(N)]), index=range(N)) for j in range(fN)})
#pairwise_plot(X_kpca, titanic["Survived"])

alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=2, min_samples_leaf=2)
alg.fit(X_kpca, titanic["Survived"])
pred = alg.predict(X_kpca)
acc = sum([prd == sv for (prd, sv) in zip(pred, titanic["Survived"])]) / float(len(pred))
print("[KernelPCA] {0:.2f}%".format(100*acc))
scores = cross_validation.cross_val_score(alg, X_kpca, titanic["Survived"], cv=10)
print("[KernelPCA-CV] {0:.2f}%".format(100*scores.mean()))

# m = fN
# algorithms = [[LinearSVC(), ['kpca'+str(i+1) for i in range(m/2)]], [KNeighborsClassifier(n_neighbors=1), ['kpca'+str(i+1) for i in range(m/2, m)]]]
# kf = KFold(X_kpca.shape[0], n_folds=10, random_state=1)
# predictions = []
# for train, test in kf:
#     beta, algs = adaboost_train(algorithms, X_kpca.iloc[train, :], titanic["Survived"].iloc[train])
#     predictions.append(adaboost_predict(beta, algs, X_kpca.iloc[test, :]))
# predictions = np.concatenate(predictions, axis=0)
# acc = sum([prd == sv for (prd, sv) in zip(predictions, titanic["Survived"])]) / float(len(predictions))
# print("[Adaboost: SVC-KNN] {0:.2f}%, bad :(".format(100*acc))

# # Generate Submission
# T_kpca = kpca.transform(titanic_test[p])
# N = len(T_kpca)
# T_kpca = pd.DataFrame({'kpca'+str(j+1): pd.Series([T_kpca[i][j] for i in range(N)], index=range(N)) for j in range(fN)})
# predictions = alg.predict(T_kpca)
# submission = pd.DataFrame({"PassengerId": titanic_test["PassengerId"], "Survived": predictions})
# submission.to_csv(SUBMISSION_PATH, index=False)
