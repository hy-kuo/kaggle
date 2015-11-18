# Kaggle
Current observation
 * dataset contains missing values
 * training, test data seem to be drawn from the same distribution
 * contains some string data
 * does data contains outliers?

## Current Progress
 * a small dataset with only 891 instances
 * fill missing value with simple mean
 * convert string data into categorical value
 * extract family_size, title from features
 * feature selection: univariant
 * prevent overfit: cross-validation

## Improvement Ideas
 * ensembling more models
 * maybe training full and missing data separately is a good try?
 * PCA/LDA?
 * if data contains outliers
  * doing data preprocessing: binning
  * change loss function
  * add constraints
 * dimension reduction
  * is there other pattern hidden in feature space?
