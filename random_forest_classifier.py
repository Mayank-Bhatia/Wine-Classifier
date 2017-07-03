# random forest classifier

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

redwine = pd.read_csv('data/wine_quality_red.csv', sep=';')

# dealing with outliers
redwine_ = redwine.loc[(redwine['fixed acidity'] > 6) & (redwine['fixed acidity'] < 13)
                       & (redwine['volatile acidity'] < 1.1)
                       & (redwine['citric acid'] < 0.7)
                       & (redwine['residual sugar'] < 5.5)
                       & (redwine['chlorides'] < 0.15) 
                       & (redwine['free sulfur dioxide'] < 38) 
                       & (redwine['total sulfur dioxide'] < 150) 
                       & (redwine['density'] > 0.992) & (redwine['density'] < 1.002)
                       & (redwine['pH'] > 3) & (redwine['pH'] < 3.7) 
                       & (redwine['sulphates'] < 1) 
                       & (redwine['alcohol'] > 8.5) & (redwine['alcohol'] < 13)]


# quality__category__meaning
#    3      0         "bad"
#    4      0         "bad"
#    5      1         "meh"  
#    6      1         "meh"
#    7      2         "good"
#    8      2         "good"
categorizer = lambda x: 0 if x == 3 else 0 if x == 4 else 1 if x == 5 else 1 if x == 6 else 2 # make categories
redwine_categorized = redwine_.assign(category = list(redwine_.quality.apply(categorizer))) # create "category" column
redwine_categorized = redwine_categorized.drop('quality', axis=1) # drop quality column


# define training and testing sets
train, test = train_test_split(redwine_categorized, test_size = 0.2) #20% split

train_y = train[train.columns[-1:]] # category label
train_y = np.reshape(train_y.values,[1005,])
train_X = train[train.columns[:-1]] # features

test_y = test[test.columns[-1:]]
test_y = np.reshape(test_y.values,[252,])
test_X = test[test.columns[:-1]]

seed = 1 # set seed


# hyperparameter optimization
clf = RandomForestClassifier(n_jobs = -1, oob_score = True, random_state = seed)

param_grid = [{'n_estimators': [200,250,300],
               'criterion': ['gini', 'entropy'],
               'max_features': ['sqrt', 'log2']}]

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, refit=True)


# train and test model
grid_search.fit(train_X, train_y)

# uncomment to see best parameters and score
# grid_search.best_params_
# grid_search.best_estimator_.score(test_X, test_y)
