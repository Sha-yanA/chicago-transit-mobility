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
        # Ensure CRS is correct for web mapping right away
        if gdf.crs != "EPSG:3857":
            gdf = gdf.to_crs(epsg=3857)

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


# --- Main Application Logic ---
def main():
    """Defines the Streamlit app structure and logic."""

    # --- Sidebar for User Inputs ---
    st.sidebar.header("Simulate a New Scenario")
    st.sidebar.markdown("""
    Adjust the sliders to represent a hypothetical census tract and predict its transit accessibility score.
    """)

    # Create sliders for each feature the model expects
    if optimal_features and gdf is not None:
        # Use min/max from the actual data for realistic slider ranges
        stop_count = st.sidebar.slider("Number of CTA Stops", int(gdf['stop_count'].min()), int(gdf['stop_count'].max()), int(gdf['stop_count'].mean()))
        service_frequency = st.sidebar.slider("Service Frequency (Avg Daily Trips)", int(gdf['service_frequency'].min()), int(gdf['service_frequency'].max()), int(gdf['service_frequency'].mean()))
        population_density = st.sidebar.slider("Population Density (per sq km)", float(gdf['population_density'].min()), float(gdf['population_density'].max()), float(gdf['population_density'].mean()))
        stop_density = st.sidebar.slider("Stop Density (stops per sq km)", float(gdf['stop_density'].min()), float(gdf['stop_density'].max()), float(gdf['stop_density'].mean()))
        pct_minority = st.sidebar.slider("Percent Minority Population", 0.0, 100.0, float(gdf['pct_minority'].mean()))
        pct_commute_public_transit = st.sidebar.slider("Percent Commuting via Public Transit", 0.0, 100.0, float(gdf['pct_commute_public_transit'].mean()))
        median_income = st.sidebar.slider("Median Household Income ($)", int(gdf['median_income'].min()), int(gdf['median_income'].max()), int(gdf['median_income'].mean()))
        avg_daily_pickups = st.sidebar.slider("Average Daily Rideshare Pickups", int(gdf['avg_daily_pickups'].min()), int(gdf['avg_daily_pickups'].max()), int(gdf['avg_daily_pickups'].mean()))

    # --- Prediction Block ---
    if st.sidebar.button("Predict Accessibility Score"):
        if model and optimal_features and gdf is not None:
            # Create input dataframe from sidebar values
            input_data = pd.DataFrame({
                'stop_count': [stop_count],
                'service_frequency': [service_frequency],
                'population_density': [population_density],
                'stop_density': [stop_density],
                'pct_minority': [pct_minority],
                'pct_commute_public_transit': [pct_commute_public_transit],
                'median_income': [median_income],
                'avg_daily_pickups': [avg_daily_pickups]
            })

            # --- START: DEBUGGING CODE ---
            st.subheader("🕵️ Debugging Info")
            st.write("**Features the Model Expects (from `optimal_features.joblib`):**")
            st.write(optimal_features)
            st.write("**Features the App Generated (from sidebar, before reordering):**")
            st.write(list(input_data.columns))
            # --- END: DEBUGGING CODE ---

            try:
                # Ensure the columns match the model's training features exactly in name and order
                input_data = input_data[optimal_features]

                # --- More Debugging ---
                st.write("**Features After Reordering (what is sent to model):**")
                st.write(list(input_data.columns))
                st.success("Feature lists seem to match and reordering was successful.")
                # --- End More Debugging ---

                # Scale the input data ONLY if the scaler was loaded
                if scaler:
                    input_data_scaled = scaler.transform(input_data)
                else:
                    input_data_scaled = input_data
                
                # Make prediction
                prediction = model.predict(input_data_scaled)
                
                st.sidebar.success(f"Predicted Accessibility Score: {prediction[0]:.3f}")

            except KeyError as e:
                st.error(f"**KeyError:** A feature is missing from the sidebar data that the model needs. The missing feature is: **{e}**")
            except Exception as e:
                st.error(f"An unexpected error occurred during prediction: {e}")


    # --- Main Panel Display ---
    st.title("Chicago Transit Accessibility & Equity Dashboard")

    st.markdown("### Interactive Accessibility Map")
    
    # --- PyDeck Map Configuration ---
    if gdf is not None:
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
    else:
        st.warning("Map data could not be loaded. Please check file paths and try again.")


if __name__ == "__main__":
    main()
