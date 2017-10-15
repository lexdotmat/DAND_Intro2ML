# Intro to Machine Learning (DAND)

==============
## Projects : Enron Dataset

This document is linked to the Intro to machine learning project. 
This project is part of Udacity Data Analyst NanoDegree and aims to use and apply machine learning techniques in a real-world use case.

This document's follows this structure:
- Chapter 1, Enron Dataset / Questions, describes the main stakes of the project.
- Chapter 2, Feature selection, describes the feature selection process and the results
- Chapter 3, Algorithm selection and tuning, describes the algorithm selection and which parameters have been chosen.
- Chapter 4, Evaluation, discusses validation and validation strategy as well as metrics.


### 1 Dataset and Questions

#### Dataset introduction

The dataset is downloaded from https://www.cs.cmu.edu/~./enron/.

The dataset is composed of 146 observation of 21 variables. There is 18 POI labelled in the dataset. 

Here is an introduction about the dataset provided by the website, please for more information visit the page Here is an introduction about the dataset provided by the website, please for more information visit the page https://www.cs.cmu.edu/~./enron/ . 

"This dataset was collected and prepared by the CALO Project (A Cognitive Assistant that Learns and Organizes). It contains data from about 150 users, mostly senior management of Enron, organized into folders. The corpus contains a total of about 0.5M messages. This data was originally made public, and posted to the web, by the Federal Energy Regulatory Commission during its investigation.
Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]"

List of variables:
['bonus', 'deferral_payments', 'deferred_income', 'director_fees', 'email_address', 'exercised_stock_options', 'expenses', 'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'loan_advances', 'long_term_incentive', 'other', 'poi', 'restricted_stock', 'restricted_stock_deferred', 'salary', 'shared_receipt_with_poi', 'to_messages', 'total_payments', 'total_stock_value']


#### Project Questions 

The project aims to identify POI (Person of Interest) based on financial data such as salary, bonuses and stocks will be used and e-mails flows.

### 2 Feature selection

#### Outliers removal :

Two Outliers were present in the data and have been removed, the first has been highlighted during the DAND lessons, the second has been found while exploring the NaN values in the dataset.

THE TRAVEL AGENCY IN THE PARK
14.29 % complete

TOTAL
71.43 % complete


- TOTAL automatically calculated from spreadsheet software
- THE TRAVEL AGENCY IN THE PARK : Present in the data and I assume, based on a the name, that it does not identify a person. More probably a business, maybe in a Park.

#### Feature engineering:

While looking at the available features, one directly pop out of the other, the director fees. After a quick internet search, I found a document explaining in detail this feature.

Report of Investigation of Enron Corporation and Related Entities Regarding Federal Tax and Compensation Issues, Etc., Volume I: Report, February 2003
 
It can  be found here:
 https://books.google.ch/books?id=yhSA2u91BFgC&pg=PA585&lpg=PA585&dq=director+Fees+enron&source=bl&ots=NcVbmM94su&sig=LXGDdcN3YJWfZfy_VNP6ZBw8Ft4&hl=de&sa=X&ved=0ahUKEwjJ-7Lu0_LWAhVCL8AKHXcBBpEQ6AEIJjAA#v=onepage&q=director%20Fees%20enron&f=false

In a few word, the 'directors fees' are special fees that concern Non-employees directors i.e. the Board of Directors. I would think that the feature is better used as a Board of Director identifier.

The feature will be used to identify Board members and non-board members.  The original feature itself will not be used in the feature used.

### 3 Algorithm selection 

What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]
What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

#### Interesting algorithms : 
Naive Bayes  http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
SVM - http://scikit-learn.org/stable/modules/svm.html 
Decision Tree :  http://scikit-learn.org/stable/modules/tree.html#classification 
Gradient Boosting classifier  - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

#### Parameter Tuning:
The following GridSearchCV as been used to setup the values for the GradientBoostingClassifier:

param_grid = {'n_estimators': [50,500], 'learning_rate': [0.05,0.1,0.2], 'criterion': ['mae'], 'max_depth': [1,2,3,4,5,6,7,8,9,10,15,40]}
clfCV= GridSearchCV(svr, param_grid)
clfCV.fit(features_train, labels_train)
print clfCV.best_params_
result : {'n_estimators': 50, 'learning_rate': 0.05, 'criterion': 'mae', 'max_depth': 1}
https://stackoverflow.com/questions/45151043/extract-best-pipeline-from-gridsearchcv-for-cross-val-predict

### 4 Evaluation


#### Evaluation: The GB classifier results: 
 
 GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.05, loss='deviance', max_depth=1,
              max_features=None, max_leaf_nodes=None,
              min_impurity_split=1e-07, min_samples_leaf=1,
              min_samples_split=2, min_weight_fraction_leaf=0.0,
              n_estimators=100, presort='auto', random_state=0,
              subsample=1.0, verbose=0, warm_start=False)
	Accuracy: 0.86480	Precision: 0.47840	Recall: 0.15500	F1: 0.23414	F2: 0.17923
	Total predictions: 15000	True positives:  310	False positives:  338	False negatives: 1690	True negatives: 12662

#### Evaluation: Standard Tree with default paramaters

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')
	Accuracy: 0.80133	Precision: 0.24690	Recall: 0.23900	F1: 0.24289	F2: 0.24054
	Total predictions: 15000	True positives:  478	False positives: 1458	False negatives: 1522	True negatives: 11542

#### Conclusion:
The Accuracy, precision of the GradientBoostingClassifier are higher than the standard decision tree algorithm.

The fitting of the classifier with GridSearchCV validated the values to be used in the final algorithm fitting.


### Resources

Along with the url you can find in the code (comments), here are some interesting page I found:

Scikit Learn Supervised learning 
http://scikit-learn.org/stable/supervised_learning.html 
Gaussian NB
http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
NB Fitting answer on SO
https://stackoverflow.com/questions/26569478/performing-grid-search-on-sklearn-naive-bayes-multinomialnb-on-multi-core-machin