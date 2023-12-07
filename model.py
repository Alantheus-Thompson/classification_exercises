import pandas as pd
import os
from pydataset import data
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import env
from sklearn.model_selection import train_test_split
import acquire as a
import prepare as p

def preprocess_titanic(train_df, val_df, test_df):
    '''
    preprocess_titanic will take in three pandas dataframes
    of our titanic data, expected as cleaned versions of this 
    titanic data set (see documentation on acquire.py and prepare.py)
    
    output:
    encoded, ML-ready versions of our clean data, with 
    columns sex and embark_town encoded in the one-hot fashion
    return: (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    '''
    # with a looping structure:
    # for df in [train_df, val_df, test_df]:
    #     df.drop(blah blah blah)
    #     df['pclass'] = df['pclass'].astype(int)
    train_df = train_df.drop(columns='passenger_id')
    train_df['pclass'] = train_df['pclass'].astype(int)
    val_df = val_df.drop(columns='passenger_id')
    val_df['pclass'] = val_df['pclass'].astype(int)
    test_df = test_df.drop(columns='passenger_id')
    test_df['pclass'] = test_df['pclass'].astype(int)
    encoding_var = ['sex', 'embark_town']
    encoded_dfs = []
    for df in [train_df, val_df, test_df]:
        df_encoded_cats = pd.get_dummies(
            df[['embark_town', 'sex']],
              drop_first=True).astype(int)
        encoded_dfs.append(pd.concat(
            [df,
            df_encoded_cats],
            axis=1).drop(columns=['sex', 'embark_town']))
    return encoded_dfs