# Training Script For Best Model

#   GradientBoostingClassifier()
#    ____    _____     _____
#   /       |     \   /  
#  |        |      | |    
#  |  ____  |_____/  |
#  |      | |     \  |
#  |      | |      | |
#   \____/  |_____/   \_____


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
_output = './data/output.csv'
_submission = './data/submission.csv'

access_token = "o.RZB1G1dB4sngsycNcfzaGIgYuU7bfJDJ"

# Read the data from file
test_data = pd.read_csv(_test)
train_data = pd.read_csv(_train)

# Remove Unnecesarry Columns
train_data = train_data.drop(['id'], axis=1)
test_data = test_data.drop(['id'], axis=1)

def BodyMassIndex():
    train_data['BMI'] = train_data['Weight'] / (train_data['Height'] ** 2)
    bmi_column = train_data.pop('BMI') 
    train_data.insert(1, 'BMI', bmi_column)

    test_data['BMI'] = test_data['Weight'] / (test_data['Height'] ** 2)
    bmi_column = test_data.pop('BMI') 
    test_data.insert(1, 'BMI', bmi_column)

BodyMassIndex() 

target = ['Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I',
       'Obesity_Type_II', 'Obesity_Type_III', 'Overweight_Level_I',
       'Overweight_Level_II']

# Encode all the data
for column in train_data.columns[:]:
    if train_data[column].dtype == 'O':
        encoder = LabelEncoder()
        train_data[column] = encoder.fit_transform(train_data[column]) + 1
        mapping_dict = dict(zip(encoder.classes_, encoder.transform(encoder.classes_) + 1))
        print(f"Mapping for {column}: {mapping_dict}")

for column in test_data.columns[:]:
    if test_data[column].dtype == 'O':
        encoder = LabelEncoder()
        test_data[column] = encoder.fit_transform(test_data[column]) + 1
        mapping_dict = dict(zip(encoder.classes_, encoder.transform(encoder.classes_) + 1))

# Split the training and testing data

rows = train_data.shape[0]
cols = train_data.shape[1]

y_data = pd.DataFrame(train_data['NObeyesdad'])
X_data = pd.DataFrame(train_data.iloc[:,:-1])
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Settings For Training The Model

param_grid = {
    'n_estimators': [540, 545, 542, 541],
    'learning_rate': [0.1, 0.06, 0.08, 0.05],
    'max_depth': [1, 2, 3],
    # 'min_samples_split': [2, 5, 10],
    # 'min_samples_leaf': [1, 2, 4],
    # 'subsample': [0.8, 0.9, 1.0]
}

start = time.time()
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
end = time.time()
model = GradientBoostingClassifier()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=4)

# Send Estimative Time To Train

lenght = 4 * 4 * 2
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

# Save the output dataframe to file

print('Saving dataframe to {_output} file . . .')
predictions = grid_search.predict(test_data)
output_data = pd.DataFrame({'NObeyesdad': [target[pred - 1] for pred in predictions]}, columns=['id', 'NObeyesdad'])
output_data['id'] = range(20758, 20758 + len(output_data))
output_data.to_csv(_output, index=False)

# Number of itterations: 245
# Time in hours: 
