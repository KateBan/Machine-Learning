#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.grid_search import GridSearchCV

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary',
                  'exercised_stock_options',
                  'bonus']
                               
#True False False False  True  True False False False  True False False
# False False False False False False False False
    
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    print len(data_dict.keys())

### Task 2: Remove outliers
del data_dict['THE TRAVEL AGENCY IN THE PARK']
del data_dict['TOTAL']

# removing NaN values and using a median value for replacement
for feature in features_list[1:]:
    feature_values = []
    for val in data_dict.values():
        if val[feature] != 'NaN':
            feature_values.append(val[feature])
    
    median = np.median(feature_values)
    
    for key in data_dict.keys():
        if data_dict[key][feature] == 'NaN':
            data_dict[key][feature] = median
            
### Task 3: Create new feature(s)
#Did not use the new features
'''
new_feature = 'bonus_salary_ratio'
features_list.append(new_feature)
for key in data_dict.keys():
    data_dict[key][new_feature] = float(data_dict[key]['bonus']/data_dict[key]['salary'])
    
    
new_feature_two = 'exercised_total_stock_ratio'
features_list.append(new_feature_two)
for key in data_dict.keys():
    data_dict[key][new_feature_two] = float(data_dict[key]['exercised_stock_options']/data_dict[key]['total_stock_value'])
'''   
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#clf = GaussianNB()
#clf = SVC()
clf = KNeighborsClassifier()
#clf = DecisionTreeClassifier()

param_grid = dict()

pipe = make_pipeline(clf)
#print pipe.named_steps.keys()

#param_grid['selectkbest__k']=[2, 3, 4, 5, 6]

#param_grid['svc__C'] = [0.001, 0.01, 0.1, 1, 10, 100]
#param_grid['svc__gamma'] = [0.001, 0.01, 0.1, 1, 10, 100]


param_grid['kneighborsclassifier__n_neighbors'] = [1, 2, 3, 4, 5, 6, 7, 8]
param_grid['kneighborsclassifier__weights'] = ['uniform', 'distance']
param_grid['kneighborsclassifier__algorithm'] = ['auto', 'ball_tree', 'kd_tree', 'brute']
param_grid['kneighborsclassifier__leaf_size'] = [2, 10, 20, 30, 36]
param_grid['kneighborsclassifier__p'] = [1,2]


#param_grid['decisiontreeclassifier__criterion'] = ['gini', 'entropy']
#param_grid['decisiontreeclassifier__max_features'] = ['auto', 'sqrt', 'log2', None]
#param_grid['decisiontreeclassifier__class_weight'] = ['auto', None]
#param_grid['decisiontreeclassifier__random_state'] = [42]

cv = StratifiedShuffleSplit(labels,test_size=0.2, random_state=42)

grid = GridSearchCV(pipe, param_grid=param_grid , cv =cv)
grid.fit(features, labels)

clf = grid.best_estimator_
#print grid.best_estimator_

#final_selectKB = grid.best_estimator_.named_steps['selectkbest']
#print final_selectKB.get_support()
#print final_selectKB.scores_

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)