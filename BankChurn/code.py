#Training Script For Best Model

import time
import numpy as np
import pandas as pd
import pickle as pkl
import datetime as dt
import warnings as wn

from pushbullet import PushBullet
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import GradientBoostingClassifier

# Ignore All Warnings
wn.filterwarnings('ignore')

# Settings For Training The Model 

_test = './data/test.csv'
_train = './data/train.csv'
_info = './model/model.log'
_model = './model/model.pkl'
_submission = './data/submission.csv'

access_token = "o.RZB1G1dB4sngsycNcfzaGIgYuU7bfJDJ"

# Read The Dataset
test = pd.read_csv(_test)
train = pd.read_csv(_train)

# Remove Unnecesarry Columns
train = train.drop(['id'], axis=1)
test = test.drop(['id'], axis=1)

# Split the training and testing data

rows = train.shape[0]
cols = train.shape[1]

y_data = pd.DataFrame(train['Exited'])
X_data = pd.DataFrame(train.iloc[:,:-1])
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Encode all the data

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

for column in X_train.columns[:]:
    if X_train[column].dtype == 'O':
        encoder = LabelEncoder()
        X_train[column] = encoder.fit_transform(X_train[column]) + 1
        mapping_dict = dict(zip(encoder.classes_, encoder.transform(encoder.classes_) + 1))

for column in X_test.columns[:]: 
    if X_test[column].dtype == 'O':
        encoder = LabelEncoder()
        X_test[column] = encoder.fit_transform(X_test[column]) + 1
        mapping_dict = dict(zip(encoder.classes_, encoder.transform(encoder.classes_) + 1))

# Settings For Training The Model

param_grid = {
    #'n_estimators': [50, 100, 200],
    #'learning_rate': [0.01, 0.1, 0.2],
    #'max_depth': [3, 4, 5],
    #'min_samples_split': [2, 5, 10],
    # 'min_samples_leaf': [1, 2, 4],
     'subsample': [0.8, 0.9, 1.0],
     'max_features': [None, 'sqrt', 'log2']
}


start = time.time()
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
end = time.time()
model = GradientBoostingClassifier()
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)

# Send Estimative Time To Train

lenght = 27 * 2
model_time = (end-start)
iterations = lenght * grid_search.cv

pb = PushBullet(access_token)
data = "Metrics For Traning"
text = [
    f'Model algorithm: {GradientBoostingClassifier()}\n\n',
    f'Model settings:  {param_grid}\n\n',
    f'Time to train one time: {model_time} sec\n',
    f'Number of iterations: {iterations}\n\n',
    f'Classification raport: {classification_report(y_test, y_pred)}\n\n'
    f'Total time to train: {model_time * iterations / 3600} Hours\n',
]
push = pb.push_note(data, ''.join(text))
print(''.join(text))
# Test On The Server For The Best Values 

start = time.time()
grid_search.fit(X_train, y_train)
grid_data = grid_search.predict(X_test)
test_accuracy = grid_search.score(X_test, y_test)
end = time.time()
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation accuracy: {:.2f}".format(grid_search.best_score_))
print("Test set accuracy: {:.2f}".format(test_accuracy))
print(f"Start time: {start}")
print(f"End time: {end}")
print("Time on training model: ", (end-start) / 3600, " Hours")

info = open(_info, "w")
model_info = [f"Best parameters found: {grid_search.best_params_}\n", 
              f"Best cross-validation accuracy: {grid_search.best_score_}\n"
              f"Time:   {dt.datetime.now()}\n\n",
              f'Classification raport: {classification_report(y_test, y_pred)}\n\n',
              f"Start time: {start}\n",
              f"End time: {end}\n",
              f"Time to train: {(end-start) / 3600} Hours\n"]
info.writelines(model_info)
pkl.dump(model, open(_model, 'wb'))
print("Save the model. . .")

# Notify On The Mobile Device 
pb = PushBullet(access_token)
data = "Model was sucessfully finished!"
push = pb.push_note(data, ''.join(model_info))

# Number of itterations: 81 x 3
# Time in hours: 
