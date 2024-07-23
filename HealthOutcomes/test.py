# The Code For Submission on Kaggle

import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
import warnings as wn

from xgboost import XGBClassifier

# Set Memory Function for data manipulation

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

# Ignore All Warnings
wn.filterwarnings('ignore')

_test = './data/test.csv'
_model = './model/model.pkl'
_submission = './data/sample_submission.csv'

# Read The Dataset
test = import_data(_test, index_col = "id", engine="pyarrow")

# Set the datatype of each column for memory eficiency

test["Region_Code"] = test["Region_Code"].astype(np.int8)
test["Policy_Sales_Channel"] = test["Policy_Sales_Channel"].astype(np.int16)

test = pd.get_dummies(test, columns = ['Gender', 'Vehicle_Age', 'Vehicle_Damage'])
test.rename(columns = {'Vehicle_Age_< 1 Year': 'Vehicle Age Less than 1 yr',
                           'Vehicle_Age_1-2 Year': 'Vehicle Age between 1 and 2 yrs',
                           'Vehicle_Age_> 2 Years': 'Vehicle Age greater than 2 yrs'
                          }, inplace = True)

print(test.head(n=10), '\n\n')

# Load the model from file

with open(_model, 'rb') as file:
    model = pkl.load(file)

# Make the predictions

index = pd.read_csv(_submission, nrows=1)['id'][0]
model_pred = model.predict(test)
submission = pd.DataFrame({
    'id': range(index, index + len(test)),
    'Response': model_pred
})
submission.to_csv(_submission, index=False)
print(submission.head(n=10))