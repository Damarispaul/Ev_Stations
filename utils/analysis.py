import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors




def prepare_df(df):
    """
    Prepares and enhances the given DataFrame containing EV charging station data 
    by performing data cleaning, feature engineering, and classification.

    Parameters:
    -----------
    df : pd.DataFrame
        A DataFrame containing electric vehicle charging station information, 
        including access patterns, availability, location, and connector types.

    Returns:
    --------
    pd.DataFrame
        A transformed DataFrame with additional features, including:
        - **Access Pattern Features**:
            - `weekday_access`: Boolean indicating if the station is available on weekdays.
            - `weekend_access`: Boolean indicating if the station is available on weekends.
            - `daily_access`: Boolean indicating if the station is available daily.
            - `total_weekly_hours`: Computed weekly operating hours.
            - `availability_class`: Categorical classification of station availability 
            ('24/7', 'High', 'Medium', 'Low', 'Unknown').
            - `is_24_7`: Boolean flag for 24/7 stations.
        - **Primary Use Case Classification**:
            - `primary_use_case`: Categorizes stations as 'Workplace', 'Commuter', 
            'Travel Corridor', 'Recreational', or 'General Purpose'.
        - **Urban vs. Rural Classification**:
            - `area_type`: Determines if a station is in an 'Urban' or 'Rural' area 
            based on station density.
        - **Connector Type Analysis**:
            - `connector_count`: Number of connector types available at the station.
            - `connector_diversity`: Ratio of available connector types.
            - `dominant_connector`: Most common connector type.

    Notes:
    ------
    - Converts state codes to uppercase for consistency.
    - Uses regex-based rules to extract access days from textual data.
    - If insufficient data points exist, density-based urban/rural classification defaults to 'Unknown'.
    - Requires latitude and longitude values for station density analysis.

    Example:
    --------
    >>> df = pd.DataFrame({
    ...     'access_days': ['daily', 'business', 'saturday', 'monday'],
    ...     'access_hours': [24, 10, 8, 12],
    ...     'latitude': [37.7749, 40.7128, 34.0522, 41.8781],
    ...     'longitude': [-122.4194, -74.0060, -118.2437, -87.6298],
    ...     'con_type1': [1, 0, 1, 0],
    ...     'con_type2': [0, 1, 0, 1]
    ... })
    >>> df = prepare_df(df)
    >>> df[['weekday_access', 'weekend_access', 'availability_class', 'area_type']]
    
    """
    print("Original shape:", df.shape)



    # Convert connector columns to proper boolean format
    connector_cols = [col for col in df.columns if col.startswith('con_')]
    for col in connector_cols:
        df[col] = df[col].astype(int).astype(bool)

    # Normalize state codes
    df['state'] = df['state'].str.upper()

    # ----------------
    # 1. Access Pattern Feature Engineering
    # ----------------

    # Parse access days
    weekdays = {'monday', 'tuesday', 'wednesday', 'thursday', 'friday'}
    weekend_days = {'saturday', 'sunday'}

    def parse_access_days(access_days):
        if isinstance(access_days, str):
            access_days = access_days.lower()
            if access_days == 'daily':
                return {'weekday_access': True, 'weekend_access': True, 'daily_access': True}
            elif access_days == 'business':
                return {'weekday_access': True, 'weekend_access': False, 'daily_access': False}
            elif access_days in weekdays:
                return {'weekday_access': True, 'weekend_access': False, 'daily_access': False}
            elif access_days in weekend_days:
                return {'weekday_access': False, 'weekend_access': True, 'daily_access': False}
            elif access_days == 'business':
                return {'weekday_access': True, 'weekend_access': False, 'daily_access': False}
        
        return {'weekday_access': False, 'weekend_access': False, 'daily_access': False}

    # Apply the function to create new columns
    access_features = df['access_days'].apply(parse_access_days).apply(pd.Series)
    df = pd.concat([df, access_features], axis=1)

    # Convert access_hours to numeric
    df['access_hours'] = pd.to_numeric(df['access_hours'], errors='coerce').fillna(0)

    # Calculate weekly hours based on access patterns
    def calculate_weekly_hours(row):
        if row['daily_access']:
            return row['access_hours'] * 7
        elif row['access_days'] == 'business':
            return row['access_hours'] * 5
        else:
            return row['access_hours']

    df['total_weekly_hours'] = df.apply(calculate_weekly_hours, axis=1)

    # ----------------
    # 2. Availability Classification
    # ----------------

    # Classify availability
    def classify_availability(hours):
        if hours >= 168:  # 24*7 = 168 hours per week
            return '24/7'
        elif hours > 80:
            return 'High'
        elif hours >= 40:
            return 'Medium'
        elif hours > 0:
            return 'Low'
        else:
            return 'Unknown'

    df['availability_class'] = df['total_weekly_hours'].apply(classify_availability)

    # Identify 24/7 stations
    df['is_24_7'] = (df['availability_class'] == '24/7').astype(int)

    # Label stations by likely primary use case
    def label_use_case(row):
        if row['ev_workplace_charging']:
            return 'Workplace'
        elif row['weekday_access'] and not row['weekend_access']:
            return 'Commuter'
        elif row['daily_access'] and row['access_hours'] >= 16:
            return 'Travel Corridor'
        elif row['weekend_access'] and not row['weekday_access']:
            return 'Recreational'
        else:
            return 'General Purpose'

    df['primary_use_case'] = df.apply(label_use_case, axis=1)

    # ----------------
    # 3. Urban/Rural Classification
    # ----------------

    # Extract coordinates for density analysis
    coords = df[['latitude', 'longitude']].dropna().values

    if len(coords) > 10:
        # Use nearest neighbors to calculate station density
        n_neighbors = min(6, len(coords))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
        distances, _ = nbrs.kneighbors(coords)
        
        # Average distance to 5 nearest stations (skip the first as it's distance to self)
        avg_distances = distances[:, 1:].mean(axis=1)
        
        # Create a DataFrame with the distances
        distance_df = pd.DataFrame({
            'avg_distance': avg_distances
        }, index=df.dropna(subset=['latitude', 'longitude']).index)
        
        # Join with original dataframe
        df = df.join(distance_df)
        
        # Classify as urban/rural based on median distance
        median_distance = df['avg_distance'].median()
        df['area_type'] = df['avg_distance'].apply(
            lambda x: 'Urban' if pd.notnull(x) and x < median_distance else 
                    'Rural' if pd.notnull(x) else 'Unknown'
        )
    else:
        print("Not enough data points for density analysis")
        df['area_type'] = 'Unknown'

    # ----------------
    # 4. Connector Availability Analysis
    # ----------------

    # Calculate connector diversity score
    df['connector_count'] = df[connector_cols].sum(axis=1)
    df['connector_diversity'] = df[connector_cols].sum(axis=1) / len(connector_cols)

    # Get dominant connector types by region
    def get_dominant_connector(row):
        if not any(row[connector_cols]):
            return 'None'
        return connector_cols[np.argmax([row[col] for col in connector_cols])].replace('con_', '')

    df['dominant_connector'] = df.apply(get_dominant_connector, axis=1)

    return df