
## Machine Learning Project

###Question 1 
Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

The goal of this project is to train a classifier that can successfully predict who is a person of interest (POI) from the Enron dataset. The Enron corporation is an example of a major corporate fraud that happened in America. As preprocessing to this project, the Enron email and financial data have been combined into a dictionary, where each key-value pair in the dictionary corresponds to one person. The dictionary key is the person's name, and the value is another dictionary, which contains the names of all the features and their values for that person. The features in the data fall into three major types, namely financial features, email features and POI labels. There are 146 key-value pairs in the dictionary, 18 of which are POIs and 128 are not POIs. Also, there are 20 original features and 2 engineered features that I did not used in my algorithm. After manually inspecting the data two outliers were removed 'TOTAL' and 'THE TRAVEL AGENCY IN THE PARK'. Also, all the NaN values were replaced by their associated medians.

###Question 2
What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]


At first I was using 3 features: 'exercised_stock_options', 'bonus' and 'total_stock_value'. I used SelectKBest for their selection. Also, it appears that for different algorithms the parameter 'k' in SelectKBest varies. In order to tune its parameters, GridSearchCV was used. Then StandardScaler was used together in a pipeline to standardize the features by removing the mean and scaling to unit variance. Two new features were engineered but none of them was picked by SelectKBest as a promising one so they were not included in the model. The first feature was 'bonus_salary_ratio' which was part of the mini project and it was looking promising, POIs are associated with high salaries and bonuses. The second one 'exercised_total_stock_ratio' which was a ratio between the exercised stock values and the total stock value the associated person had. POIs exercised their stocks before the company crashed.

The estimated scores for the features from the SelectKbest selection function respectively are [2.74248681e+01, 1.59796067e+01, 2.36775927e+01]

Then I removed SelectKbest, StandardScaler and I manually set the features to be 'exercised stock options', 'salary' and 'bonus' and all of this made the evaluation metrics to rise significantly. The reason I chose those metrics is the documentary movie I watched 'Enron: The smartest Guys in the room'. There it was shown that people like Ken Lay And Jeff Skilling had exercised their stock options before the bankruptcy of the company, they also had high salaries and bonuses which we saw in the mini project.

###Question 3
What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

The algorithm I ended up using is KNeighborsClassifier. As listed below is gave the best accuracy, precision and recall. The values below are for the model without using StandardScaler and SelectKBest.
 - KNN: Accuracy: 0.90260	Precision: 0.83562	Recall: 0.33550
Compared to the values when StandardScaler and SelectKBest were used.
 - KNN: Accuracy: 0.87787	Precision: 0.68919	Recall: 0.15300
 - SVC: Accuracy: 0.86747	Precision: 0.51230	Recall: 0.12500 
 - DTC: Accuracy: 0.79893	Precision: 0.21299	Recall: 0.18850 


###Question 4
What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]

To tune the parameters of an algorithm means to select the best possible combination of parameters values that gives you the best possible performance. If you don't do that the defaulst will be used which may not give you the best result. In my case I used GridSearchCV to tune the parameters. For the KNN algorithm the parameters are as follows:

 - param_grid['kneighborsclassifier__n_neighbors'] = [1, 2, 3, 4, 5, 6, 7, 8]
 - param_grid['kneighborsclassifier__weights'] = ['uniform', 'distance']
 - param_grid['kneighborsclassifier__algorithm'] = ['auto', 'ball_tree', 'kd_tree', 'brute']
 - param_grid['kneighborsclassifier__leaf_size'] = [2, 10, 20, 30, 36]
 - param_grid['kneighborsclassifier__p'] = [1,2]
 
 
###Question 5
What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]

Validating is separating your dataset into training and testing dataset. This way you can train your algorithm on the training dataset and test how well it performs on an unknown data by using the testing dataset. The data separation should be randomized in a way that the testing and training sets are balanced. In our case, it is important to separate the data in a way that it did not end up with training data set consisting just of non-POIs and testing with POIs, for example.
I used StratifiedShuffleSplit cross validation iterator that provides train/test indices to split data in train test sets. This cross-validation object is a merge of StratifiedKFold and ShuffleSplit, which returns stratified randomized folds. The folds are made by preserving the percentage of samples for each class.

###Question 6
Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

In our case the number of POIs is much higher than non-POIs, therefore, predicting that everyone is not a POI will give much higher accuracy and this is not useful. Precision and recall are much better metrics, they capture the percentage of correct identified POIs. Precision is about 0.84 which means that 84% of the identified POIs are actually POIs. Recall is the number of identified POIs out of all the known POIs which is equal to 34%.

###References
 - The sklearn documentation
 - Introduction to sklearn: https://www.youtube.com/playlist?list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A
 - O'Reilly - Advanced Machine Learning with scikit learn
 - Udacity materials and forums
 -  “I hereby confirm that this submission is my work. I have cited above the origins of any parts of the submission that were taken from Websites, books, forums, blog posts, github repositories, etc."



    
