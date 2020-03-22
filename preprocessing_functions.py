#!/usr/bin/env python
# coding: utf-8

# In[ ]:


############## PREPORCESSING FUNCTIONS ###########################
import pandas as pd
import numpy as np

def unique_count(dataframe):
    import pandas as pd
    #shows how many unique values are in the columns of a dataframe
    uniques = dataframe.from_records([(col, dataframe[col].nunique()) for col in dataframe.columns], 
                                        columns=['Column_Name', 'Num_Unique']).sort_values(by=['Num_Unique'])
    uniques.reset_index(inplace=True)
    return uniques

def DeleteMissing(dataframe, perc_missing_delete = 0.6):
    import pandas as pd
    return_df = dataframe[dataframe.columns[dataframe.isnull().mean() <= perc_missing_delete]]
    return return_df

def get_column_names_from_ColumnTransformer(column_transformer):   
    import pandas as pd
    col_name = []
    for transformer_in_columns in column_transformer.transformers_[:-1]:#the last transformer is ColumnTransformer's 'remainder'
        raw_col_name = transformer_in_columns[1]
        if isinstance(transformer_in_columns[1],Pipeline): 
            transformer = transformer_in_columns[1].steps[-1][1]
        else:
            transformer = transformer_in_columns[1]
        try:
            names = transformer.get_feature_names()
        except AttributeError: # if no 'get_feature_names' function, use raw column name
            names = raw_col_name
        if isinstance(names,np.ndarray): # eg.
            col_name += names.tolist()
        elif isinstance(names,list):
            col_name += names    
        elif isinstance(names,str):
            col_name.append(names)
    return col_name

def Imputer_df(dataframe, num_cols_names, cat_cols_names, strategy_num_cols = 'mean',
               strategy_cat_cols = 'most_frequent', remainder = 'drop'):    
    import pandas as pd
    import numpy as np
    #imputes missing values using SimpleImputer from sklearn and returns a dataframe instead of numpy array
    
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer

    names = num_cols_names+cat_cols_names+[name for name in dataframe.columns if name not in num_cols_names+cat_cols_names]
    ct1 = ColumnTransformer([
        ('imput_num', SimpleImputer(strategy = strategy_num_cols), num_cols_names),
        ('imput_cat', SimpleImputer(strategy = strategy_cat_cols), cat_cols_names)
        ], remainder = 'passthrough')
    return_df = pd.DataFrame(data = ct1.fit_transform(dataframe), columns = names)
    return return_df


def ColumnValuesCheck(df1, df2):
    # Columns in both dataframes must have the same names
    #checks if columns with the same names have the same values. Columns do not have to be in the same order
    #DO NOT USE AFTER IMPUTING MISSING VALUES
    import pandas as pd
    list = []
    for column in df1:
        column_values1 = df1[column]
        column_values2 = df2[column]
        if column_values1.equals(column_values2):
            list = list + [1]
        else:
            list = list + [0]
    return list


def MakeDummiesAndScale(data, cat_cols, num_cols):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    for cat in enumerate(cat_cols):
        #add category names to the values - this way get_dummies will create columns with names
        data[cat[1]] = data[cat[1]].apply(lambda x : str(cat[1]) +  '_' + str(x))
        
        #get dummies
        dummy = pd.get_dummies(data[cat[1]])
        
        #select all column except the last one
        dummy = dummy.iloc[:,0:dummy.shape[1]-1].copy()
        
        data = data.merge(dummy, left_index = True, right_index = True)
        #alternatively data = pd.concat([data, dummy], axis = 1)
    #remove original columns from the dataset
    data.drop(cat_cols, axis = 1, inplace = True)
    
    sc = StandardScaler()
    data[num_cols] = sc.fit_transform(data[num_cols])
    
    return data

def train_test_split_df(data_X, y, test_size = 0.2, random_state = 0):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data_X, y, test_size = 0.2, random_state = 0)
    y_train = pd.DataFrame(data = y_train, columns = ['default3'])
    y_test = pd.DataFrame(data = y_test, columns = ['default3'])
    return X_train, X_test, y_train, y_test

