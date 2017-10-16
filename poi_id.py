#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# SECTION 1:  DATASET and QUESTIONS
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Dataset exploration

# Import to a dataframe and transpose to have a tidy format:
df = pd.DataFrame(data_dict)
df = df.T
# Count the number of observation:
#  Count the number of variable:
print "Dataset size:"
print df.shape[0], "observations"
print df.shape[1], "variables"
print
# print available variables
print "List of variables:"
print list(df)
print
# POI:
print "Number of POI in the data:", (df['poi'] == True) .sum()

print
print "variable completeness"
print
print "Number of NaN Values", df.loc[:, df.isnull().any()]
for column in df:
    print column
    full_col = (df[column] != 'NaN').sum()
    total_col = df[column].count()
    print round((float(full_col)/total_col)*100,2) , "% complete"
    print
    # https://stackoverflow.com/questions/10768724/why-does-python-return-0-for-simple-division-calculation
    # Note: Add the float to get result

# https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas

print
print "Observation completeness"
print

# remove observation which are inferior to 20 % completeness
for index, row in df.iterrows():
    full_row = (df.loc[index]!= 'NaN').sum()
    total_row = df.shape[1]
    completeness = round((float(full_row)/total_row)*100,2)
    if completeness < 20:
        print index, "removed"
        print completeness, "% complete"
        print df.loc[index]['poi']
        data_dict.pop (index ,0)
    # remove outlier found in the courses

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# SECTION 2:  Outliers removal
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Remove the TOTAL Outlier

# 1: TOTAL from spreadsheets tools
data_dict.pop('TOTAL', 0 )
# 2: Input which is not a person
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0 )
# 3: only 2% of the data points present in the data:
data_dict.pop('LOCKHART EUGENE E', 0 )

for key, value in data_dict.items():
    if value['director_fees'] != 'NaN':
        value[ 'is_director' ] = True
        print "Is member of board of Director: ", key
    else:
        value[ 'is_director' ] = False

# FINAL LIST OF FEATURES:
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# features_list = [ 'poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
        #                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
        #         'restricted_stock', 'director_fees', 'is_director', 'to_messages', 'from_poi_to_this_person', 'from_messages',
    #         'from_this_person_to_poi', 'shared_receipt_with_poi']

features_list = [ 'poi',  'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'expenses', 'exercised_stock_options', 'other',
                 'restricted_stock', 'director_fees', 'is_director', 'to_messages', 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi']
# features_list = [ 'poi', 'salary', 'deferral_payments', 'bonus',
#                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
#                 'restricted_stock',  'to_messages', 'from_poi_to_this_person', 'from_messages',
#                 'from_this_person_to_poi', 'shared_receipt_with_poi']
#features_list = [ 'poi',  'total_payments', 'bonus', 'is_director' ,
#                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
#                 'restricted_stock','shared_receipt_with_poi']


# Excluded email features:
# Text string: 'email_address', 'to_messages' 'from_messages'

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# SECTION 3:  Selection and Creation of New feaures
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# New feature based on the director fees
# Store to my_dataset for easy export below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()

data = featureFormat(my_dataset, features_list, sort_keys = True)

from imblearn.over_sampling import SMOTE, ADASYN
labels, features = targetFeatureSplit(data)
features_resampled, labels_resampled = SMOTE().fit_sample(features, labels)
# features = scaler.fit_transform(features)
# features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 0)

features_train, features_test, labels_train, labels_test = train_test_split(features_resampled, labels_resampled, test_size = 0.3, random_state = 0)

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
print "RandomForestClassifier "
clf_Ranking = RFECV(GradientBoostingClassifier(random_state=0, learning_rate= 0.05, max_depth=1), scoring='accuracy', n_jobs = -1)
clf_Ranking.fit(features_train, labels_train)
print clf_Ranking.score(features_train, labels_train)
print clf_Ranking.ranking_
# result of feature selection : [ 1 13  4 14  1 12 11  8  1  9  5  6  1  2 10  7  3  1]
# [1 4 5 1 1 1 1 1 3 1 1 1 6 2 1 1 1 1]
# [14  5  1 11  1 10  4  1  1  1  6  3  2  9  8 12 13  7  1]
#print scores

# GBC : [13 12 11 10  3  1  1  9  1  1  1  8  1  7  6  2  4  5  1  1]


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# SECTION 4:  Classifier Selection
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.svm import SVC
#param_grid = [
#  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
#]
treeclf = tree.DecisionTreeClassifier()
SVMclf = svm.SVC( C = 0.1, gamma = 1000)
NBclf = GaussianNB()
Gbclf = GradientBoostingClassifier(random_state=0)

treeclf.fit(features_train, labels_train)
SVMclf.fit(features_train, labels_train)
NBclf.fit(features_train, labels_train)
Gbclf.fit(features_train, labels_train)

y_pred_tree = treeclf.predict(features_test)
y_pred_SVM  = SVMclf.predict(features_test)
y_pred_NB   = NBclf.predict(features_test)
y_pred_GB   = Gbclf.predict(features_test)

#param_grid = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}
#svr =  tree.DecisionTreeClassifier(random_state = 0)
#clf = GridSearchCV(svr, param_grid, verbose=1)

# svr = GradientBoostingClassifier(random_state=0)
# svr0 = GradientBoostingClassifier(random_state=0)
# param_grid = {'n_estimators': [50,500], 'learning_rate': [0.05,0.1,0.2], 'criterion': ['mae'], 'max_depth': [1,2,3,4,5,6,7,8,9,10,15,40]}
# clfCV= GridSearchCV(svr, param_grid)
# clfCV.fit(features_train, labels_train)
# print clfCV.best_params_
# result : {'n_estimators': 50, 'learning_rate': 0.05, 'criterion': 'mae', 'max_depth': 1}
# https://stackoverflow.com/questions/45151043/extract-best-pipeline-from-gridsearchcv-for-cross-val-predict

clf = GradientBoostingClassifier(random_state=0, learning_rate= 0.1,max_depth=2, max_features = 'auto', n_estimators = 500)
#clf = GaussianNB()
#clf = RandomForestClassifier()
clf.fit(features_train, labels_train)

y_pred = clf.predict(features_test)


print 'Number of training points', len(labels_train)
print 'Number of test points', len(labels_test)

print "SVM Report"
print classification_report(labels_test, y_pred_SVM)
print "NB Report"
print classification_report(labels_test, y_pred_NB)
print "Tree Report"
print classification_report(labels_test, y_pred_tree)
print "Gradient Boosted Tree Report"
print classification_report(labels_test, y_pred_GB)
#print "optimized Tree Report"
print classification_report(labels_test, y_pred)



### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/

#Import libraries:


# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)