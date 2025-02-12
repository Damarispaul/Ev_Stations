#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def outliers(data_frame):
    """ Function to check for outliers in each column and return their value counts"""
    col_names= []
    col_count = []
    for col in data_frame.columns:
        # Calculate the first and third quartiles (Q1 and Q3)
        Q1 = data_frame[col].quantile(0.25)
        Q3 = data_frame[col].quantile(0.75)
        # Calculate the interquartile range (IQR)
        IQR = Q3 -Q1
        # Define the lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # find the value beyound the lower and upper bounds
        outliers = data_frame[col][(data_frame[col] < lower_bound) |
                           (data_frame[col] > upper_bound)]
        #append the column names and tye number of outliers
        col_names.append(col)
        col_count.append(outliers.value_counts().sum())
    return list(zip(col_names,col_count))
   


# Function to remove the outliers
def Remove_Outliers(dataframe):
    """
    A function that checks for the outliers beyond the bounds in each column and removes rows with those outliers
    then returns the new data frame without the outliers.
    """
    # Make a copy of the dataframe to avoid modifying the original dataframe
    df_cleaned = dataframe.copy()
    
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype in ['int64', 'float64']:
            # Calculate the first and third quartiles (Q1 and Q3)
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)

            # Calculate the interquartile range (IQR)
            IQR = Q3 - Q1

            # Define the lower and upper bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Filter out the rows with outliers in the current column
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
        else: pass

    # Return the shape of the cleaned dataframe and the dataframe itself
    return df_cleaned

