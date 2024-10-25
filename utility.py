#!/usr/bin/env python
# coding: utf-8

# # Utility!
# 
# A colleciton of a useful funcitons that I've found on Kaggle over the years or wrote myself to avoid future work and keep my notebooks a bit more readable.
# 
# If it was something I found on kaggle, I put credits in each cell of where I found them... LMK if I have the original authors wrong.

# In[1]:


from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
from IPython.display import display_html
import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import cross_val_score, cross_validate, KFold


# In[2]:


def calculate_mean_encodings(train, target, categorical_columns, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = []  # Initialize an empty list if no columns are to be excluded

    encoding_list = []  # Use a list to collect all DataFrames    
    
    # Loop through categorical columns and calculate mean target per category
    for col in categorical_columns:
        if col not in exclude_columns:  # Skip the column if it's in the exclude list
            # Calculate the mean target per category, treating NaNs as a separate category
            category_means = train.groupby(col, dropna=False)[target].mean().reset_index()
            # Rename columns for clarity
            category_means.rename(columns={col: 'category', target: 'mean_target'}, inplace=True)
            # Add a column to identify the original column name
            category_means['col_name'] = col
            # Adjust columns order
            category_means = category_means[['col_name', 'category', 'mean_target']]
            encoding_list.append(category_means)
    
    # Concatenate all into a single DataFrame
    if encoding_list:
        encodings = pd.concat(encoding_list, ignore_index=True)
    else:
        encodings = pd.DataFrame(columns=['col_name', 'category', 'mean_target'])  # Return empty DataFrame if no encodings
    
    return encodings


    

def apply_encodings_and_replace(data, encodings):
    """
    Applies mean target encodings to the categorical columns of the dataset and replaces the original
    columns with the encoded ones by creating new columns and then dropping the old ones.
    
    Args:
    data (DataFrame): The dataset to which encodings should be applied.
    encodings (DataFrame): DataFrame containing the encodings, expected to have 'col_name', 'category', and 'mean_target' columns.

    Returns:
    DataFrame: The dataset with the original categorical columns replaced by their encoded values.
    """
    encoded_data = data.copy()
    for col in data.columns:
        if col in encodings['col_name'].unique():
            # Create a new column for the encoded values
            encoded_column_name = col + '_encoded'
            encoded_data[encoded_column_name] = np.nan
            # Fetch the relevant encodings for this column
            column_encodings = encodings[encodings['col_name'] == col]
            for _, row in column_encodings.iterrows():
                category = row['category']
                mean_target = row['mean_target']
                # Assign the mean target value to the new column wherever the categories match
                encoded_data.loc[encoded_data[col] == category, encoded_column_name] = mean_target
            
            # Optionally handle categories not found in the encodings (unseen categories)
            if encoded_data[encoded_column_name].isna().any():
                # Fill NaNs with the overall target mean as calculated in the def above
                encoded_data[encoded_column_name].fillna(encodings.iat[-1, -1], inplace=True)

            # Drop the original column after encoding is complete
            encoded_data.drop(col, axis=1, inplace=True)
            # Optionally rename the encoded column to the original name if needed
            encoded_data.rename(columns={encoded_column_name: col}, inplace=True)

    return encoded_data


# In[3]:


#size of CSV/parquet/etc in a directory
def get_disk_usage(directory):
    cmd = f'du {directory}/* -h | sort -rh'
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, text=True)
    output_lines = result.stdout.split('\n')

    # Extract file/directory names and sizes
    data = [line.split('\t') for line in output_lines if line]
    df = pd.DataFrame(data, columns=['size', 'path'])
    df['file_name'] = df.path.str.replace('train_|test_', '', regex=True).    apply(lambda x: Path(x).stem)
    return df
'''
train_disk_usage = get_disk_usage(f'{ROOT}/csv_files/train').reset_index()
test_disk_usage = get_disk_usage(f'{ROOT}/csv_files/test')

train_disk_usage.reset_index().merge(test_disk_usage, on=['file_name'],
                                     how='outer', suffixes=['_train', '_test'])\
                                     .sort_values(by='index').drop(columns=['index'])
'''


# In[4]:


# https://www.kaggle.com/code/darynarr/enefit-online-training/notebook

# feature forward selection
# change params to take in a model, and maybe a DF
def select_columns(is_consumption = True):
    selected_columns = pd.Series({"year": 1e5, "month": 1e5, "day":1e5, 'hour':1e5})
    
    while True:
        n_selected = len(selected_columns)
        current_loss = selected_columns.min()
        for col in X.columns:
            if col in selected_columns.index:
                continue
            cols = selected_columns.index.tolist() + [col]
            
            res = cross_validate(
                # rewrite this for imput of model
                estimator=get_model(
                    params['consumption'] if is_consumption else params['production'], 
                    verbose=False, cat_cols = [c for c in cat_cols if c in cols]),
                X=X[X['is_consumption']==int(is_consumption)][cols],
                y=y[X['is_consumption']==int(is_consumption)],
                scoring="neg_mean_absolute_error",
                cv= cv,
                return_estimator=True
            )
            current_loss = -res['test_score'].mean()
            
            if current_loss < selected_columns.min():
                selected_columns[col] = current_loss
                logger.info(f"{len(selected_columns)}, {col}, {current_loss}")
        if n_selected == len(selected_columns):
            break
    return selected_columns


# In[5]:


# https://www.kaggle.com/code/kimtaehun/breif-eda-and-xgb-baseline-with-full-dataset
def summary(df):
    print(f'data shape: {df.shape}')
    summ = pd.DataFrame(df.dtypes, columns=['data type'])
    summ['#missing'] = df.isnull().sum().values 
    summ['%missing'] = df.isnull().sum().values / len(df) * 100
    summ['#unique'] = df.nunique().values
    desc = pd.DataFrame(df.describe(include='all').transpose())

    # Debug: print the columns of desc
    # print("Columns in desc:", desc.columns)
    
    if 'min' in desc.columns:
        summ['min'] = desc['min'].values
    else:
        summ['min'] = 'N/A'

    if 'max' in desc.columns:
        summ['max'] = desc['max'].values
    else:
        summ['max'] = 'N/A'
    
    summ['first value'] = df.iloc[0].values if len(df) > 0 else 'N/A'
    summ['second value'] = df.iloc[1].values if len(df) > 1 else 'N/A'
    summ['third value'] = df.iloc[2].values if len(df) > 2 else 'N/A'    

#    return summ
    display_html(summ)
    print("\n")


# In[6]:


# From chatGPT and then me editng it
def ohe_categorical(df, categorical_cols, threshold=10, keep_col = True):
    """
    One-hot encodes the categorical columns in the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        categorical_cols (list): List of column names to one-hot encode.
        threshold (int): The threshold for the number of unique values
                         in a column to decide whether to one-hot encode it.

    Returns:
        pd.DataFrame: The DataFrame with one-hot encoded columns.
        list: List of columns that weren't one-hot encoded.
    """

    # Find the columns that weren't one-hot encoded
    not_encoded_cols = []

    # Iterate through the categorical columns
    for col in categorical_cols:
        unique_count = df[col].nunique()
        if unique_count <= threshold:
            # Perform one-hot encoding for columns with unique_count less than or equal to the threshold
            if keep_col: copied_col = df[col]
            df = pd.get_dummies(df, columns=[col])
            if keep_col: df[col] = copied_col
            
        else:
            # Append the column name to the not_encoded_cols list
            not_encoded_cols.append(col)

    return df, not_encoded_cols


# In[7]:


def get_columns_by_type(df, data_type):
    columns_list = []
    
    for column in df.columns:
        if df[column].dtype == data_type:
            columns_list.append(column)
    
    return columns_list


# In[8]:


# https://www.kaggle.com/code/andradaolteanu/rsna-fracture-detection-dicom-images-explore
def df_info(df, name="Default"): 
    print(clr.S+f"=== {name} ==="+clr.E)

    if not hasattr(df, 'shape'):
        print(clr.S + "Shape:" + clr.E, format(df.shape[0], ","), format(df.shape[1], ","))
    else:
        print(clr.S + "Shape:" + clr.E, df.shape)

    print(clr.S+f"Missing Values:"+clr.E, format(df.isna().sum().sum(), ","), "total missing datapoints.")
    print(clr.S+"Columns:"+clr.E, list(df.columns), "\n")
    
    display_html(df.tail())
    print("\n")

class clr:
    S = '\033[1m' + '\033[94m'
    E = '\033[0m'
    
my_colors = ["#5EAFD9", "#449DD1", "#3977BB", 
             "#2D51A5", "#5C4C8F", "#8B4679",
             "#C53D4C", "#E23836", "#FF4633", "#FF5746"]
CMAP1 = ListedColormap(my_colors)


# In[9]:


def read_and_display_csv(directory, date_field):
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.csv'):
                print(os.path.join(dirname, filename))
                no_ext = f'{os.path.splitext(filename)[0]}_df'
                no_ext = no_ext.replace(" ", "_")
                no_ext = no_ext.replace("-", "_")
                
                # Read the CSV headers first to check for the date_field
                temp_df = pd.read_csv(os.path.join(dirname, filename), nrows=0)
                if date_field in temp_df.columns:
                    # Parse dates if the date_field is present, but don't set it as the index
                    globals()[no_ext] = pd.read_csv(
                        os.path.join(dirname, filename), 
                        parse_dates=[date_field],
                        index_col=False  # Ensure the date field is not set as the index
                    )
                else:
                    # Read normally without parsing dates
                    globals()[no_ext] = pd.read_csv(os.path.join(dirname, filename), index_col=False)
                del temp_df
                df_info(globals()[no_ext], no_ext)


# In[10]:


# These helper and data cleaning functions are from the old fast.ai course
# The repository is here: https://github.com/fastai/fastai/tree/master/old
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
        
def make_date(df, date_field:str):
    "Make sure `df[field_name]` is of the right date type."
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)
        


# In[11]:


def add_datepart(df, fldnames, drop=True, time=False, errors="raise", sin_cos=False, exclude_cols=[]):
    """
    Add Date Parts converts a column of df from a datetime64 to many columns containing 
    the information from the date. It returns a modified version of the original DataFrame.
    If sin_cos is True, sine and cosine features for the relevant datetime components
    are also added.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the date column.
    fldnames (str or list): The name(s) of the date column(s).
    drop (bool): Whether to drop the original date column(s).
    time (bool): Whether to include time-related date parts (Hour, Minute, Second).
    errors (str): How to handle parsing errors when converting to datetime.
    sin_cos (bool): Whether to add sine and cosine features for date components.
    exclude_cols (list): List of columns to exclude from date part creation.

    Returns:
    pd.DataFrame: The DataFrame with added date parts.
    """

    def create_sine_cosine_features(df, column, max_value):
        if column in df.columns:
            df[f'{column}_sin'] = np.sin(2 * np.pi * df[column] / max_value)
            df[f'{column}_cos'] = np.cos(2 * np.pi * df[column] / max_value)
        return df

    if isinstance(fldnames, str):
        fldnames = [fldnames]
    
    for fldname in fldnames:
        if fldname in exclude_cols:
            continue  # Skip this column if it's in the exclude list
            
        fld = df[fldname]
        fld_dtype = fld.dtype
        
        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            fld_dtype = np.datetime64

        if not np.issubdtype(fld_dtype, np.datetime64):
            df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)
        
        targ_pre = re.sub('[Dd]ate$', '', fldname)
        attr = ['Year', 'Month', 'Day', 'Dayofweek', 'Dayofyear',
                'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 
                'Is_year_end', 'Is_year_start']
        
        if time:
            attr = attr + ['Hour', 'Minute', 'Second']
        
        for n in attr:
            df[targ_pre + n] = getattr(fld.dt, n.lower())
        
        if drop:
            df = df.drop(fldname, axis=1)

        if sin_cos:
            # Sine and Cosine transformations
            df = create_sine_cosine_features(df, targ_pre + 'Hour', 24)
            df = create_sine_cosine_features(df, targ_pre + 'Day', 30)  # Approximation
            df = create_sine_cosine_features(df, targ_pre + 'Dayofweek', 7)
            df = create_sine_cosine_features(df, targ_pre + 'Dayofyear', 365)  # May need adjustment for leap years
            df = create_sine_cosine_features(df, targ_pre + 'Month', 12)
            # Add additional sine and cosine features for Minute and Second if required
            if time:
                df = create_sine_cosine_features(df, targ_pre + 'Minute', 60)
                df = create_sine_cosine_features(df, targ_pre + 'Second', 60)

    return df


# In[12]:


import numpy as np

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:




