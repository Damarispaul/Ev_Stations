#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Check for the dataset information   
def check_info(df):
    """Displays dataset shape, columns, data types, missing values, duplicates, and description."""
    
    print("=============== Dataset Shape =================")
    print(df.shape)
    
    print("\n=============== Dataset Columns =================")
    print(df.columns)
    
    print("\n=============== Data Types =================")
    print(df.dtypes)
    
    print("\n=============== Dataset Information =================")
    df.info()  
    print("\n=============== Concise Statistic Summary =================")
    print(df.describe())
    
    print("\n=============== Missing Values =================")
    print(df.isnull().sum())
    
    print("\n=============== Duplicated Rows =================")
    print(df.duplicated().sum())
    
   
    
def outliers(data_frame):
    """Checks for outliers in numerical columns using the IQR method."""
    
    col_names = []
    col_count = []
    
    for col in data_frame.select_dtypes(include=['int64', 'float64']).columns:  # Process only numeric columns
        # Calculate Q1, Q3, and IQR
        Q1 = data_frame[col].quantile(0.25)
        Q3 = data_frame[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers = data_frame[col][(data_frame[col] < lower_bound) | (data_frame[col] > upper_bound)]

        # Append results
        col_names.append(col)
        col_count.append(outliers.count())  # Use `.count()` instead of `.value_counts().sum()`
    
    return list(zip(col_names, col_count))


def remove_outliers(dataframe):
    """Removes outliers in numerical columns based on the IQR method."""
    
    df_cleaned = dataframe.copy()  # Avoid modifying original data

    for col in df_cleaned.select_dtypes(include=['int64', 'float64']).columns:  # Process only numeric columns
        # Calculate Q1, Q3, and IQR
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out outliers
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    
    return df_cleaned  # Return cleaned DataFrame
