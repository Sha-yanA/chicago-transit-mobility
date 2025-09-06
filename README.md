# Evaluating Urban Mobility in Chicago with Machine Learning

 A complete, end-to-end Machine Learning pipeline to evaluate urban mobility in Chicago. The pipeline covers everything from acquiring the data to analyzing the results for efficiency and equity. Check it out on https://chicago-transit-mobility.streamlit.app

## 1. Project Framing and Goal Definition

This project aims to create a quantitative "Transit Accessibility Score" for each census tract in Chicago. This score will represent the ease of accessing essential services and opportunities via public and private transportation.

- Efficiency Analysis: How well does the current transit system serve the population in different areas?

- Equity Analysis: Is there a disparity in the Transit Accessibility Score across different socio-economic and racial groups? For example, do low-income or predominantly minority neighborhoods have lower scores?

The final model will predict this score based on various features, allowing us to identify which factors (demographic, infrastructural) most significantly impact transit accessibility.

## 2. Data Acquisition and Ingestion

This step involves gathering all the necessary datasets. We will use a common geographic identifier, the census tract, to merge these disparate sources.

Public Transit Data (CTA):
- GTFS Data: The GTFS feed from the CTA Developer Center. This contains multiple .txt files (stops.txt, routes.txt, trips.txt, stop_times.txt) that define the structure of the transit network.
- Ridership Data: Download the "CTA - Ridership - Daily Boarding Totals" dataset from the Chicago Data Portal.

Ride-Hailing Data (TNP):
 -  Transportation Network Providers - Trips" dataset from the Chicago Data Portal. This contains aggregated trip data, including pickup/dropoff census tracts.

Demographic & Geospatial Data:
- Census Data: Use the U.S. Census Bureau API to fetch data from the American Community Survey (ACS) for all census tracts in Cook County, Illinois. Key variables to pull include:

            Total Population (B01003_001E)

            Median Household Income (B19013_001E)

            Population by Race (e.g., White, Black, Asian, Hispanic)

            Number of households with no vehicle (B08201_002E)
            
- Shapefiles: Download TIGER/Line shapefiles for Chicago census tracts to enable geospatial analysis and visualization.

## 3. Data Preprocessing and Cleaning

This is a critical step to standardize and merge the data. We use libraries like Pandas and GeoPandas to clean and arrange the data to make it workable.

 1. Geospatial Alignment:
- Load the census tract shapefile into a GeoDataFrame. This will be our base geographical unit.
- Ensure all other datasets (TNP, Census) have a column for the census tract FIPS code that can be used for merging.

2. GTFS Data Processing:
- Clean stop locations: Filtering out non-operational stops and ensure coordinates are in a standard format (WGS 84).
- Spatially join the transit stops (stops.txt) to the census tract shapefile to determine which tract each stop falls into.

 3. TNP & Ridership Data Cleaning:
 - Convert date/time columns to datetime objects.
- Handle missing values. For trip data, you might drop rows with missing location information.
- Aggregate the data:
 
		Calculate the average daily TNP pickups and drop-offs for each census tract.
			
		Calculate the average daily CTA boardings for each station/stop and then aggregate these to the census tract level.

4. Census Data Cleaning:
- Rename cryptic API column names (e.g., B19013_001E) to be human-readable (e.g., Median_Income).
- Handle any missing data, often represented by negative values in census data. Imputation or removal might be necessary.

## 4. Exploratory Data Analysis (EDA) and Feature Engineering

Now we explore the data and engineer the features that will power our model. The goal is to create a rich set of predictors for each census tract.

### Exploratory Data Analysis
- Mapping: Create choropleth maps to visualize key variables.

	 - Map the density of CTA bus stops and 'L' stations.

	- Map the median household income by census tract.

	- Map the percentage of different racial groups by census tract.

	- Map the average number of TNP pickups. This will reveal mobility patterns.

- Correlation Analysis: Use a heatmap to check for correlations between variables. For example, is there a negative correlation between CTA ridership and median income in certain areas?

### Feature Engineering

For each census tract, we will calculate the following features:

- Transit Supply Features:

        stop_count: Total number of CTA bus and train stops.

        stop_density: Number of stops per square kilometer.

        service_frequency: Average number of bus/train arrivals per hour (especially during peak hours, e.g., 7-9 AM). This requires a complex analysis of the stop_times.txt and trips.txt files.

        unique_routes: Number of unique bus/train routes serving the tract.

- TNP & Other Mobility Features:

        avg_tnp_pickups: Average daily TNP pickups originating in the tract.

        pct_no_vehicle_households: Percentage of households with no available vehicle (from Census data).

- Demographic Features (from Census API):

        population_density: Population per square kilometer.

        median_income: Median household income.

        pct_minority: Percentage of the population that identifies as non-white.

        pct_poverty: Percentage of the population below the poverty line.

## 5. Model Selection and Training

Since we don't have a pre-existing "accessibility score," we'll first use unsupervised learning to discover natural groupings of census tracts and then build a composite index for a supervised model.

### Unsupervised Clustering (to define accessibility tiers)

1. Algorithm Choice: Use K-Means Clustering. It's effective at partitioning data into a predefined number (k) of groups based on feature similarity.

2. Preparation:

	- Scale all numerical features using StandardScaler from scikit-learn to ensure that variables with larger ranges don't dominate the clustering process.

	- Use the "Elbow Method" or "Silhouette Score" to determine the optimal number of clusters (k). Let's assume we find k=4.

3. Execution: Train the K-Means model on your engineered features. This will assign each census tract to one of the 4 clusters.

4. Interpretation: Analyze the feature characteristics of each cluster. You might find they naturally correspond to labels like:

	- Cluster 0: High density, high frequency, high income (e.g., "High-Access Urban Core").

	- Cluster 1: Low density, low frequency, low income (e.g., "Transit Deserts").

	- Cluster 2: Moderate density, rail-focused, middle income (e.g., "Suburban-Style Access").

	- Cluster 3: High density, bus-dependent, lower-middle income.

### Supervised Learning (to create a predictive score)

1. Create the Target Variable (Accessibility_Score):

	- Based on the clustering results and domain knowledge, create a composite index. Normalize key supply features (like stop_density and service_frequency) to a 0-1 scale and combine them.

	- Formula Example: Accessibility_Score = 0.5×(norm_stop_density)+0.5×(norm_service_frequency)

	- This score is now our ground truth target variable (y).

2. Algorithm Choice: Use a Gradient Boosting Regressor like XGBoost or LightGBM or a Random Forest. These models are powerful, handle complex interactions between features, and can provide feature importance scores.

3. Training:
	- Split the data into training (80%) and testing (20%) sets.
	- Train the model to predict the Accessibility_Score using the demographic and TNP features as predictors (X).

## 6. Model Evaluation

We need to evaluate both the clustering and regression models.

- Clustering Evaluation:

	- Silhouette Score: A value close to +1 indicates that tracts are well-matched to their own cluster and poorly-matched to neighboring clusters.

	- Visual Validation: Plot the cluster assignments on a map of Chicago. Do the clusters form geographically coherent regions that make intuitive sense?

- Regression Evaluation:
	-	Metrics: On the test set, calculate the following:

            R-squared (R2): The proportion of the variance in the Accessibility_Score that is predictable from the features. An R2 of 0.75 means 75% of the score's variance is explained by the model.

            Mean Absolute Error (MAE): The average absolute difference between the predicted and actual scores. It's easily interpretable.

	- Feature Importance: Plot the feature importance from the trained model. This is crucial for the equity analysis.

## 7. Interpretation and Equity Analysis

The final and most important step, where we draw conclusions.

1. Visualize the Score: Create a choropleth map of Chicago showing the final predicted Accessibility_Score for every census tract.

 2. Cross-Tabulation: Compare the average Accessibility_Score across different demographic groups.
	- Group census tracts by their majority racial group and compare the average scores.

	- Create bins for median income (e.g., <$30k, $30-60k, >$60k) and compare the average scores.

3. Answer the Core Question: Use the feature importance plot and the cross-tabulations to draw concrete conclusions. For example:
	- "Our model shows that median_income and pct_minority are significant negative predictors of the Accessibility Score, indicating that lower-income, higher-minority neighborhoods have systematically lower access to high-quality transit."

	- "The map reveals a clear transit divide between the North Side and the South and West Sides of Chicago."

## 8. Deployment and Real-time Prediction 

To make your model interactive and forward-looking, you can deploy it.

- API Development: Wrap your trained regression model in a web API using Flask or FastAPI.

- Endpoint: Create an endpoint that accepts a census tract ID as input.

- Functionality: When the endpoint is called:
	1. It retrieves the static features for that tract (demographics, etc.).
	2. For real-time adjustments, it could call the CTA Bus Tracker API to check for current service disruptions or delays near that tract.
	3. It feeds these features into the loaded model and returns the predicted Accessibility_Score as a JSON response.

- Frontend: Build a simple web dashboard (using Streamlit or Dash) that displays the map of accessibility scores and allows users (like city planners or residents) to click on a tract to see its score and underlying data.

## 9. Monitoring and Maintenance

A model is a living asset and must be maintained.

- Data Updates: The CTA GTFS data, ridership figures, and ACS estimates are updated periodically. Schedule a quarterly or biannual pipeline run to ingest new data.

- Model Retraining: Retrain the model with the new data to ensure its predictions remain accurate as the city changes.

- Performance Monitoring: Keep track of the model's evaluation metrics (R2, MAE) over time to detect any performance degradation, known as model drift.Evaluating Urban Mobility in Chicago with Machine Learning
