import streamlit as st
import pandas as pd
import geopandas as gpd
import joblib
import pydeck as pdk
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Chicago Transit Equity Analysis",
    layout="wide",
)


# --- Caching Data Loading ---
@st.cache_data
def load_data():
    """
    Loads all necessary files for the app: GeoJSON data, the ML model,
    the list of optimal features, and the scaler (if it exists).
    """
    try:
        gdf = gpd.read_file("chicago_accessibility_predictions.geojson")
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs(epsg=4326)

        model = joblib.load("transit_accessibility_model.joblib")
        optimal_features = joblib.load("optimal_features.joblib")

        # Safely try to load the scaler; it might not exist if SVR wasn't the best model
        scaler = None
        if os.path.exists("svr_scaler.joblib"):
            scaler = joblib.load("svr_scaler.joblib")
            
        return gdf, model, scaler, optimal_features
        
    except FileNotFoundError as e:
        st.error(f"ERROR: A required data file was not found: {e.filename}. Please make sure all necessary files from the notebook are in the same directory as this app.")
        return None, None, None, None

# --- Load Data ---
gdf, model, scaler, optimal_features = load_data()

# --- Main App Logic (runs only if data is loaded) ---
if gdf is not None and model is not None and optimal_features is not None:
    
    # --- Sidebar ---
    st.sidebar.title("Urban Mobility Explorer")
    st.sidebar.markdown("Analyze transit accessibility and equity across Chicago's neighborhoods.")

    # --- Real-Time Prediction Section (Now Dynamic) ---
    st.sidebar.header("Real-Time Accessibility Prediction")
    st.sidebar.markdown("Adjust the sliders below for the features your model selected as most predictive.")

    # A dictionary to hold the configuration for each possible slider
    slider_configs = {
        'median_income': {"label": "Median Household Income ($)", "min": 10000, "max": 250000, "default": 60000, "step": 1000},
        'pct_minority': {"label": "Percent Minority Population", "min": 0.0, "max": 100.0, "default": 25.0, "step": 0.5},
        'pct_commute_public_transit': {"label": "Percent Commuting via Public Transit", "min": 0.0, "max": 100.0, "default": 15.0, "step": 0.5},
        'population_density': {"label": "Population Density (per sq/km)", "min": 500, "max": 30000, "default": 5000, "step": 100},
        'avg_daily_pickups': {"label": "Avg. Daily Rideshare Pickups", "min": 0, "max": 500, "default": 50, "step": 5}
    }
    
    # A dictionary to store the user's input values
    user_inputs = {}

    # Dynamically create sliders based on the loaded optimal_features
    for feature in optimal_features:
        if feature in slider_configs:
            config = slider_configs[feature]
            user_inputs[feature] = st.sidebar.slider(
                config["label"], 
                min_value=config["min"], 
                max_value=config["max"], 
                value=config["default"], 
                step=config["step"]
            )

    # --- Prediction Logic ---
    if st.sidebar.button("Predict Score", use_container_width=True):
        # Create a dataframe from the dynamic inputs
        input_data = pd.DataFrame([user_inputs])
        
        # Ensure the column order matches the model's training order
        input_data = input_data[optimal_features]
        
        # Scale the input data ONLY if the scaler was loaded (i.e., SVR was the best model)
        if scaler:
            input_data_scaled = scaler.transform(input_data)
        else:
            input_data_scaled = input_data

        input_data_scaled = input_data_scaled[optimal_features]
        print(f"hellooooo... {input_data_scaled}")
        
        # Make the prediction
        predicted_score = model.predict(input_data_scaled)
        
        # Display the result
        st.sidebar.metric("Predicted Accessibility Score", f"{predicted_score:.3f}")
        
        # Interpret the score based on our cluster analysis
        if predicted_score >= 0.7:
            st.sidebar.success("This score corresponds to an **'A: High-Access Corridor'** area.")
        elif predicted_score >= 0.4:
            st.sidebar.info("This score corresponds to a **'B: Moderate-Access Neighborhood'**.")
        elif predicted_score >= 0.2:
            st.sidebar.warning("This score corresponds to a **'C: Low-Access Area'**.")
        else:
            st.sidebar.error("This score corresponds to a **'D: Transit Desert'**.")

    # --- Main Page Content ---
    st.title("Chicago Transit Accessibility & Equity Dashboard")

    st.markdown("### Interactive Accessibility Map")
    
    # --- PyDeck Map Configuration ---
    view_state = pdk.ViewState(latitude=41.8781, longitude=-87.6298, zoom=9, pitch=45, bearing=0)

    layer = pdk.Layer(
        "GeoJsonLayer",
        gdf,
        opacity=0.6,
        stroked=True,
        filled=True,
        extruded=True,
        wireframe=True,
        get_elevation="accessibility_score * 2000",
        get_fill_color="[255 * (1 - accessibility_score), 255 * accessibility_score, 0, 140]", # Green = High, Red = Low
        get_line_color=[255, 255, 255],
        pickable=True
    )
    
    tooltip = {
        "html": """
            <b>Tract ID:</b> {tract_id} <br/>
            <b>Accessibility Tier:</b> {cluster_label} <br/>
            <b>Accessibility Score:</b> {accessibility_score:.3f} <br/>
            <b>Median Income:</b> ${median_income:,.0f} <br/>
            <b>Percent Minority:</b> {pct_minority:.1f}% <br/>
        """
    }

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/light-v9',
        tooltip=tooltip
    )

    st.pydeck_chart(r)
    st.markdown("The map visualizes each census tract in Cook County. **Height and color** both represent the calculated accessibility score—higher, greener areas have better access.")
    st.info("Hover over a tract to see its detailed stats.")

