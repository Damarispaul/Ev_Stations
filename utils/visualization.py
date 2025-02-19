import  matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster, HeatMap
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd


def create_analysis_visualizations(df):
    """
    Generates various visualizations to analyze charging station data.

    This function creates multiple plots to examine different aspects of the dataset, including availability, 
    weekly operational hours, area types, network providers, connector types, and primary use cases. 
    Each visualization is displayed and saved as an image file.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing charging station data with relevant columns, including:
        - 'availability_class' (categorical): Availability classification of stations.
        - 'total_weekly_hours' (numeric): Weekly operational hours.
        - 'area_type' (categorical): Urban or Rural classification.
        - 'ev_network' (categorical): Network provider of the station.
        - 'primary_use_case' (categorical): Use case classification.
        - 'state' (categorical): U.S. state in which the station is located.
        - Various connector type columns indicating station capabilities.

    Visualizations Created:
    -----------------------
    1. **Availability distribution**: A countplot showing the number of stations in each availability class.
    2. **Weekly hours by state**: A bar plot displaying the top 10 states with the highest average weekly hours.
    3. **Urban vs Rural availability**: A stacked bar chart comparing availability across area types.
    4. **Network provider comparison**: A bar plot showing average weekly hours by major EV network providers.
    5. **Connector availability heatmap**: A heatmap indicating the availability of connector types across different station classes.
    6. **Weekly hours distribution**: A histogram showing the distribution of weekly operational hours across all stations.
    7. **Primary use case vs Weekly hours**: A boxplot comparing weekly hours for different primary use cases.

    Outputs:
    --------
    - Displays all visualizations in a grid layout.
    - Saves each plot as an image file in PNG format for further analysis.

    Example Usage:
    --------------
    >>> create_analysis_visualizations(df)

    Notes:
    ------
    - The function assumes that the dataset includes the required columns.
    - Subplots are automatically adjusted for readability.
    - If certain categories are missing in the dataset, the function adjusts dynamically.
    """

    connector_cols = [col for col in df.columns if col.startswith('con_')]
    # Set up the style
    sns.set(style="whitegrid")
    
    # Create a figure with subplots
    fig, axes = plt.subplots(4, 2, figsize=(20, 24))
    axes = axes.flatten()  # Flatten the 4x2 grid into a 1D array for easier indexing
    
    # 1. Availability distribution
    ax = sns.countplot(y='availability_class', data=df, 
                       order=[cat for cat in ['24/7', 'High', 'Medium', 'Low', 'Unknown']
                             if cat in df['availability_class'].unique()], 
                       palette='viridis', ax=axes[0])
    axes[0].set_title('Distribution of Charging Station Availability')
    axes[0].set_xlabel('Number of Stations')
    axes[0].set_ylabel('Availability Class')
    
    # Add percentage labels
    total = len(df)
    for p in ax.patches:
        width = p.get_width()
        percentage = f'{width/total*100:.1f}%'
        ax.text(width + 5, p.get_y() + p.get_height()/2, f'{int(width)} ({percentage})', 
                va='center')
    
    # Save individual plot
    plt.figure()
    sns.countplot(y='availability_class', data=df, 
                  order=[cat for cat in ['24/7', 'High', 'Medium', 'Low', 'Unknown']
                        if cat in df['availability_class'].unique()], 
                  palette='viridis')
    plt.title('Distribution of Charging Station Availability')
    plt.xlabel('Number of Stations')
    plt.ylabel('Availability Class')
    plt.tight_layout()
    plt.savefig('availability_distribution.png')
    plt.close()
    
    # 2. Weekly hours by state (top 10 states)
    state_hours = df.groupby('state')['total_weekly_hours'].agg(['mean', 'count'])
    state_hours = state_hours[state_hours['count'] > 10].sort_values('mean', ascending=False)
    
    sns.barplot(x=state_hours.index[:10], y='mean', data=state_hours.reset_index()[:10], 
                palette='Blues_d', ax=axes[1])
    axes[1].set_title('Average Weekly Operational Hours by State (Top 10)')
    axes[1].set_xlabel('State')
    axes[1].set_ylabel('Average Weekly Hours')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Save individual plot
    plt.figure()
    sns.barplot(x=state_hours.index[:10], y='mean', data=state_hours.reset_index()[:10], 
                palette='Blues_d')
    plt.title('Average Weekly Operational Hours by State (Top 10)')
    plt.xlabel('State')
    plt.ylabel('Average Weekly Hours')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('weekly_hours_by_state.png')
    plt.close()
    
    # 3. Urban vs Rural availability comparison
    urban_rural_pivot = pd.crosstab(df['area_type'], df['availability_class'], 
                                    normalize='index') * 100
    urban_rural_pivot = urban_rural_pivot[[cat for cat in ['24/7', 'High', 'Medium', 'Low', 'Unknown']
                                          if cat in df['availability_class'].unique()]]
    urban_rural_pivot.plot(kind='bar', stacked=True, colormap='viridis', ax=axes[2])
    axes[2].set_title('Availability Classes by Area Type')
    axes[2].set_xlabel('Area Type')
    axes[2].set_ylabel('Percentage')
    axes[2].legend(title='Availability')
    
    # Save individual plot
    plt.figure()
    urban_rural_pivot.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Availability Classes by Area Type')
    plt.xlabel('Area Type')
    plt.ylabel('Percentage')
    plt.legend(title='Availability')
    plt.tight_layout()
    plt.savefig('urban_rural_availability.png')
    plt.close()
    
    # 4. Network comparison (top networks)
    network_counts = df['ev_network'].value_counts()
    top_networks = network_counts[network_counts > 20].index
    network_hours = df[df['ev_network'].isin(top_networks)].groupby('ev_network')['total_weekly_hours'].mean().sort_values(ascending=False)
    
    sns.barplot(x=network_hours.index, y=network_hours.values, palette='Blues_d', ax=axes[3])
    axes[3].set_title('Average Weekly Hours by Network Provider')
    axes[3].set_xlabel('Network')
    axes[3].set_ylabel('Average Weekly Hours')
    axes[3].tick_params(axis='x', rotation=90)
    
    # Save individual plot
    plt.figure()
    sns.barplot(x=network_hours.index, y=network_hours.values, palette='Blues_d')
    plt.title('Average Weekly Hours by Network Provider')
    plt.xlabel('Network')
    plt.ylabel('Average Weekly Hours')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('network_comparison.png')
    plt.close()
    
    # 5. Heatmap: Connector types by availability class
    connector_by_availability = df.groupby('availability_class')[connector_cols].mean()
    sns.heatmap(connector_by_availability, annot=True, fmt='.2f', cmap='YlGnBu', ax=axes[4])
    axes[4].set_title('Connector Availability by Station Class')
    axes[4].set_ylabel('Availability Class')
    axes[4].set_xlabel('Connector Type')
    
    # Save individual plot
    plt.figure()
    sns.heatmap(connector_by_availability, annot=True, fmt='.2f', cmap='YlGnBu')
    plt.title('Connector Availability by Station Class')
    plt.ylabel('Availability Class')
    plt.xlabel('Connector Type')
    plt.tight_layout()
    plt.savefig('connector_by_availability.png')
    plt.close()
    
    # 6. Weekly hours distribution
    sns.histplot(data=df, x='total_weekly_hours', bins=20, kde=True, ax=axes[5])
    axes[5].set_title('Distribution of Weekly Operational Hours')
    axes[5].set_xlabel('Weekly Hours')
    axes[5].set_ylabel('Number of Stations')
    axes[5].axvline(x=168, color='red', linestyle='--', label='24/7 (168 hours)')
    axes[5].axvline(x=40, color='orange', linestyle='--', label='Standard Workweek (40 hours)')
    axes[5].legend()
    
    # Save individual plot
    plt.figure()
    sns.histplot(data=df, x='total_weekly_hours', bins=20, kde=True)
    plt.title('Distribution of Weekly Operational Hours')
    plt.xlabel('Weekly Hours')
    plt.ylabel('Number of Stations')
    plt.axvline(x=168, color='red', linestyle='--', label='24/7 (168 hours)')
    plt.axvline(x=40, color='orange', linestyle='--', label='Standard Workweek (40 hours)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('weekly_hours_distribution.png')
    plt.close()
    
    # 7. Use case by access hours
    sns.boxplot(x='primary_use_case', y='total_weekly_hours', data=df, ax=axes[6])
    axes[6].set_title('Weekly Hours by Primary Use Case')
    axes[6].set_xlabel('Primary Use Case')
    axes[6].set_ylabel('Weekly Hours')
    axes[6].axhline(y=168, color='red', linestyle='--', label='24/7 (168 hours)')
    axes[6].legend()
    
    # Save individual plot
    plt.figure()
    sns.boxplot(x='primary_use_case', y='total_weekly_hours', data=df)
    plt.title('Weekly Hours by Primary Use Case')
    plt.xlabel('Primary Use Case')
    plt.ylabel('Weekly Hours')
    plt.axhline(y=168, color='red', linestyle='--', label='24/7 (168 hours)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('hours_by_use_case.png')
    plt.close()
    
    # Hide the last unused subplot (if any)
    if len(axes) > 7:
        axes[7].axis('off')
    
    # Adjust layout and save
    fig.subplots_adjust(hspace=0.8, wspace=0.3)
    plt.tight_layout()
    fig.savefig('charging_station_analysis.png')
    plt.show();

def create_map_visualization(df):
    # Filter out rows without coordinates
    map_df = df.dropna(subset=['latitude', 'longitude'])
    
    # Create a base map centered on the mean coordinates
    center_lat = map_df['latitude'].mean()
    center_lon = map_df['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4, 
                  tiles='CartoDB positron')
    
    # Define color scheme for availability classes
    color_dict = {
        '24/7': 'darkgreen',
        'High': 'green',
        'Medium': 'orange',
        'Low': 'red',
        'Unknown': 'gray'
    }
    
    # Create a marker cluster
    marker_cluster = MarkerCluster().add_to(m)
    
    # Add markers for each station
    for idx, row in map_df.iterrows():
        popup_text = f"""
        <b>Network:</b> {row['ev_network']}<br>
        <b>Weekly Hours:</b> {row['total_weekly_hours']}<br>
        <b>Access Days:</b> {row['access_days']}<br>
        <b>Hours per Day:</b> {row['access_hours']}<br>
        <b>Area Type:</b> {row['area_type']}<br>
        <b>Use Case:</b> {row['primary_use_case']}<br>
        <b>Connectors:</b> {row['connector_count']}
        """
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            popup=folium.Popup(popup_text, max_width=300),
            color=color_dict[row['availability_class']],
            fill=True,
            fill_color=color_dict[row['availability_class']],
            fill_opacity=0.7
        ).add_to(marker_cluster)
    
    # Add a heatmap layer for 24/7 stations
    heatmap_data = map_df[map_df['is_24_7'] == 1][['latitude', 'longitude']].values
    if len(heatmap_data) > 0:
        HeatMap(heatmap_data, radius=15, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}, 
                name="24/7 Station Density").add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save the map
    m.save('ev_charging_access_map.html')
    print("Map visualization saved as 'ev_charging_access_map.html'")
    return m
