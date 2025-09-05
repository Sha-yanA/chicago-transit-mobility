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
        st.error(f"ERROR: A required data file was not found: {e.filename}. Please make sure all necessary files from the notebook are in the same directory as the app.")
        return None, None, None, None

# --- Main App ---
def main():
    gdf, model, scaler, optimal_features = load_data()

    if gdf is None:
        st.warning("App cannot start because data files are missing.")
        return

    st.title("🏙️ Chicago Transit Accessibility & Equity Dashboard")

    st.markdown("### Interactive Accessibility Map")
    
    # --- PyDeck Map Configuration ---\
    # Tooltip configuration for hover information
    tooltip = {
        "html": """
            <b>Tract ID:</b> {tract_id} <br/>
            <b>Accessibility Tier:</b> {cluster_label} <br/>
            <b>Accessibility Score:</b> {accessibility_score:.3f} <br/>
            <b>Median Income:</b> ${median_income:,.0f} <br/>
            <b>Percent Minority:</b> {pct_minority:.1f}% <br/>
        """
    }
    
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
    
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/light-v9"
    )
    
    st.pydeck_chart(r)

    # --- Sidebar for User Input & Predictions ---
    st.sidebar.title("Accessibility Predictor")
    st.sidebar.markdown("""
        Curious about how a new development or policy change might impact accessibility? 
        Input the characteristics of a hypothetical census tract below to get a predicted 
        accessibility score.
    """)

    # --- Create Input Fields ---
    user_inputs = {}
    user_inputs['bus_routes_in_tract'] = st.sidebar.slider("Number of Bus Routes", 0, 20, 5)
    user_inputs['bus_stops_in_tract'] = st.sidebar.slider("Number of Bus Stops", 0, 50, 10)
    user_inputs['total_bus_service_hours'] = st.sidebar.slider("Total Weekly Bus Service Hours", 0, 5000, 1000)
    user_inputs['l_stops_in_tract'] = st.sidebar.slider("Number of 'L' Train Stops", 0, 5, 0)
    user_inputs['dist_to_closest_l_stop_km'] = st.sidebar.number_input("Distance to Nearest 'L' Stop (km)", min_value=0.0, max_value=20.0, value=2.5, step=0.1)
    user_inputs['total_l_service_hours'] = st.sidebar.slider("Total Weekly 'L' Service Hours", 0, 2500, 0)
    user_inputs['median_income'] = st.sidebar.number_input("Median Household Income", min_value=10000, max_value=250000, value=60000, step=1000)
    user_inputs['pct_minority'] = st.sidebar.slider("Percent Minority Population", 0.0, 100.0, 50.0, 0.5)

    # --- Prediction Logic ---
    if st.sidebar.button("Predict Accessibility Score"):
        # Convert user inputs into a pandas DataFrame
        input_df = pd.DataFrame([user_inputs])

        # Ensure all necessary feature columns are present.
        # This loop adds any features the model expects that aren't in the user input form.
        for col in optimal_features:
            if col not in input_df.columns:
                input_df[col] = 0
                
        # --- FIX: Ensure feature order and names match the model's training data ---
        # Create a DataFrame with columns explicitly ordered by `optimal_features`.
        # This is the crucial step to prevent the ValueError.
        final_input_df = input_df[optimal_features]

        # Data scaling (if a scaler was used during training)
        if scaler:
            # The scaler.transform method returns a NumPy array, which discards feature names.
            input_scaled_np = scaler.transform(final_input_df)
            
            # The model, however, likely expects a DataFrame with specific feature names.
            # We must convert the scaled array back to a DataFrame with the correct columns.
            input_for_model = pd.DataFrame(input_scaled_np, columns=optimal_features)
            
            prediction = model.predict(input_for_model)
        else:
            # For models that don't need scaling, we still pass the correctly ordered DataFrame.
            input_for_model = final_input_df
            prediction = model.predict(input_for_model)

        st.sidebar.subheader("Predicted Score:")
        st.sidebar.info(f"{prediction[0]:.3f}")

        # --- Interpretation of the score ---
        # Find the cluster label for the predicted score based on the predicted value
        score = prediction[0]
        if score > 0.7:
             cluster_label = "High-Access Hubs"
        elif score > 0.45:
            cluster_label = "Well-Connected Areas"
        elif score > 0.2:
            cluster_label = "Moderate-Access Zones"
        else:
            cluster_label = "Low-Access Areas"

        st.sidebar.metric(label="Predicted Accessibility Tier", value=cluster_label)


if __name__ == "__main__":
    main()
