#Training Script For Best Model

import time
import numpy as np
import pandas as pd
import pickle as pkl
import datetime as dt
import warnings as wn

from xgboost import XGBClassifier
from pushbullet import PushBullet
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

# Set Memory Function for efficient data manipulation

def reduce_mem_usage(df):
    """ 
        Iterate through all the columns of a dataframe and modify the 
        data type to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def import_data(file, **kwargs):
    """
        create a dataframe and optimize its memory usage
    """
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True, **kwargs)
    df = reduce_mem_usage(df)
    return df

def remove_outliers(df):
    """
        Check for outliers in the dataframe
    """
    outliers_columns = []
    total_rows = len(df)
    
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        outlier_count = outliers.sum()
        outlier_percentage = (outlier_count / total_rows) * 100
        
        if outlier_percentage >= 20:
            outliers_columns.append(column)
            print(f"Outliers detected in column '{column}': {outlier_percentage:.2f}% of total rows.")
            
            df = df[~outliers]
            print(f"Removed {outlier_count} outliers from column '{column}'.")
    
    if not outliers_columns:
        print("No columns with outliers exceeding 20% detected.")

    return df
    
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
train = import_data(_train, index_col = "id", engine="pyarrow")

# Set the datatype of each column for memory eficiency

train["Region_Code"] = train["Region_Code"].astype(np.int8)
train["Policy_Sales_Channel"] = train["Policy_Sales_Channel"].astype(np.int16)
train = remove_outliers(train)
train.drop_duplicates()

# One Hot Encode all the data

train = pd.get_dummies(train, columns = ['Gender', 'Vehicle_Age', 'Vehicle_Damage'])
train.rename(columns = {'Vehicle_Age_< 1 Year': 'Vehicle Age Less than 1 yr',
                           'Vehicle_Age_1-2 Year': 'Vehicle Age between 1 and 2 yrs',
                           'Vehicle_Age_> 2 Years': 'Vehicle Age greater than 2 yrs'
                          }, inplace = True)

print(train.head(n=10))

# Split the training and testing data

rows = train.shape[0]
cols = train.shape[1]

response = train.pop('Response')
train.insert(len(train.columns), 'Response', response)
y_data = pd.DataFrame(train['Response'])
X_data = pd.DataFrame(train.iloc[:,:-1])
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Settings For Training The Model

param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],      # Number of trees
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],   # Step size shrinkage
    'max_depth': [3, 4, 5, 6, 7],                   # Maximum depth of a tree
    'min_child_weight': [1, 3, 5, 7],               # Minimum sum of instance weight (hessian) needed in a child
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],         # Subsample ratio of the training instances
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],  # Subsample ratio of columns when constructing each tree
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],               # Minimum loss reduction required to make a further partition on a leaf node
    'reg_alpha': [0, 0.01, 0.1, 1, 10],             # L1 regularization term on weights
    'reg_lambda': [0.01, 0.1, 1, 10, 100],          # L2 regularization term on weights
    'scale_pos_weight': [1, 2, 5, 10]               # Balancing of positive and negative weights
}

start = time.time()
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
end = time.time()
model = XGBClassifier()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

number_of_combinations = len(list(ParameterGrid(param_grid)))
total_folds = number_of_combinations * grid_search.cv
model_time = (end-start)
iterations = total_folds

pb = PushBullet(access_token)
data = "Metrics For Traning"
text = [
    f'Model algorithm: {XGBClassifier()}\n\n',
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
model = XGBClassifier(**grid_search.best_params_)
model.fit(X_data, y_data)
pkl.dump(model, open(_model, 'wb'))
print("Save the model. . .")

# Notify On The Mobile Device 
pb = PushBullet(access_token)
data = "Model was sucessfully finished!"
push = pb.push_note(data, ''.join(model_info))
 
