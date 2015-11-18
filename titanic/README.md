# Experiment
Cross-Validation 10 fold
 1. [Adaboost with DecisionTree](1st_AdaBoost.py): `82.72%`
 2. [Voting: KNN+GradientBoostingClassifier](2nd_Ensembling.py): `82.83%`
 3. [Adaboost with Perceptron-RandomForest-GradientBoosting-LinearSVM-KNN](2nd_Ensembling.py): `82.04%`
 4. [KernelPCA with RandomForest](2nd_Ensembling.py): `81.04%`

## Features
 * Survival: (Integer: 0 = No; 1 = Yes)
 * PClass: (Integer: 1 = 1st; 2 = 2nd; 3 = 3rd social-economic status) Passenger Class
 * Name: (String)
 * Sex: (String: 'female', 'male')
 * Age: (Integer)
 * SibSp: (Integer: number of siblings/spouses abroad)
 * Parch: (Integer: number of parents/children abroad)
 * Ticket: (String)
 * Fare: (Integer)
 * Cabin: (String)
 * Embarked: (Character: 'C' = Cherbourg; 'Q' = Queenstown; 'S; = Southampton)

### Additional Features
 * FamilySize: (Integer: SibSp+Parch)
 * Title: (Categorized String)
