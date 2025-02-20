# Ev_Stations

![Image URL](https://i.dailymail.co.uk/1s/2022/02/16/15/54273773-0-image-a-33_1645025868934.jpg)

### 1. Business Understanding 

#### 1.1. Backgound Information & Overview

Kenya is undergoing a transportation and energy transformation, with electric vehicle (EV) adoption increasing due to rising fuel costs, government incentives, and a global push for sustainability efforts. However, the absence of a data-driven approach to charging station placement is slowing down EV adoption. Currently, charging station deployment is largely arbitrary, reactive, or limited to a few locations, leading to underutilization, range anxiety, and inefficient infrastructure investment.

##### Problem Statement

The adoption of electric vehicles (EVs) in Kenya is increasing, but the absence of a well-planned, optimized EV charging infrastructure remains a major barrier to widespread adoption. Current charging stations are placed without data-driven insights, leading to low utilization rates, inconvenient locations, and poor return on investment for operators.


##### Proposed Solution

By integrating machine learning, geospatial analytics, and optimization models, this AI-driven platform will revolutionize EV infrastructure planning in Kenya. The solution ensures that charging stations are placed where they are most needed, cost-effective, and energy-efficient, paving the way for a sustainable and profitable EV ecosystem.

* Using K-Means Clustering, DBSCAN, and Hierarchical Clustering to help map out the best possible station locations based on geography and infrastructure constraints.
* Use Graph-based Routing and Dijkstra’s Algorithm to ensure stations are placed within an optimal travel distance for EV users. For example ensuring no driver needs to travel more than 5 km to find a charging station.
* Use Random Forest Regression, XGBoost, and Gradient Boosting Machines (GBM) to identify the key drivers of charging station demand based on traffic volume, population density, nearby commercial hubs, weather conditions and charging station accessibility.

#####  Objectives

###### Primary Objective:

To develop an AI-powered platform that uses machine learning, geospatial data, and predictive analytics to identify optimal locations for electric vehicle (EV) charging stations.

###### Secondary Objectives:

1. To integrate machine learning algorithms for analyzing EV usage patterns and infrastructure needs.
2. To incorporate geospatial data for determining the most efficient and accessible sites for charging stations.
3. To ensure the platform is scalable and adaptable to various regions or cities.
4. To improve the overall efficiency of EV charging infrastructure deployment and usage.

The platform will enable:

* EV charging network planners to maximize utilization and profitability by selecting high-demand locations.
* Government agencies to accelerate green mobility initiatives through data-backed decision-making.
* Investors to make informed funding decisions, ensuring high ROI.
* EV users to access conveniently located charging stations, improving the overall user experience.

##### Metrics of Success

1. The model should correctly predict high-demand locations, minimizing false positives and negatives when identifying optimal sites.

### 2. Data Understanding

The **Alternative Fueling Stations dataset** provides information on fueling stations across the United States that offer alternative fuels such as biodiesel, electricity, ethanol, hydrogen, natural gas, and propane. The dataset is maintained by the **National Renewable Energy Laboratory (NREL)** and updated daily.

## General Columns
The dataset includes the following general attributes:

- **Station Name**: The name of the fueling station.
- **Access Code**: Indicates whether the station is **public** (accessible to all) or **private** (restricted access).
- **Access Days & Time**: Describes the hours of operation for the station.
- **Cards Accepted**: Specifies payment methods accepted at the station.
- **Street Address**: The physical address of the station.
- **City, State, ZIP Code**: Location details for filtering and mapping.
- **Latitude, Longitude**: Geographic coordinates for spatial analysis.
- **Station Phone**: Contact number for inquiries.
- **Date Last Confirmed**: The most recent date the station's data was verified.
- **Status Code**: Indicates whether the station is currently **available**, **planned**, or **temporarily unavailable**.

## EV-Specific Columns
For **electric vehicle (EV) charging stations**, the dataset includes:

- **EV Charger Type**: Specifies the charger type (e.g., **Level 1, Level 2, DC Fast**).
- **EV Connector Types**: Lists supported connectors (e.g., **CHAdeMO, CCS, Tesla**).
- **EV Network**: Identifies whether the station is part of a network (e.g., **ChargePoint, Tesla, Electrify America**).
- **EV Pricing**: Provides pricing details for using the station.
- **EV On-Site Renewable Source**: Indicates if the station is powered by renewable energy sources.

## Data Source
The data is collected in partnership with **Clean Cities coalitions** to help fleets and consumers find alternative fueling stations. The dataset is part of the **U.S. Department of Transportation (USDOT)/Bureau of Transportation Statistics (BTS) National Transportation Atlas Database (NTAD)**.

For more details, visit the [Data link](https://data-usdot.opendata.arcgis.com/datasets/alternative-fueling-stations/explore
).
