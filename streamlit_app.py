import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Define categories and thresholds for both models
grape_categories = ["Black Rot", "ESCA", "Healthy", "Leaf Blight"]
pomegranate_categories = ["Healthy_Pomogranate", "Cercospora", "Bacterial_Blight", "Anthracnose"]

grape_thresholds = {
    "Black Rot": 0.1,
    "ESCA": 0.1,
    "Healthy": 0.1,
    "Leaf Blight": 0.1
}

pomegranate_thresholds = {
    "Healthy_Pomogranate": 0.01,
    "Cercospora": 0.1,
    "Bacterial_Blight": 0.5,
    "Anthracnose": 0.1
}

# Information about each disease
disease_info = {
    "Black Rot": "Black Rot is a fungal disease that affects grapevines. It causes dark spots on leaves and shriveled berries.",
    "ESCA": "ESCA is a complex disease affecting grapevines. Symptoms include leaf discoloration and dieback.",
    "Leaf Blight": "Leaf Blight causes lesions on grapevine leaves, leading to browning and leaf drop.",
    "Cercospora": "Cercospora leaf spot affects pomegranate leaves, causing round spots that can lead to defoliation.",
    "Bacterial_Blight": "Bacterial Blight causes water-soaked lesions on pomegranate leaves and fruit.",
    "Anthracnose": "Anthracnose causes dark, sunken spots on pomegranate fruit, affecting its quality."
}

# Function to apply thresholds
def apply_thresholds(predictions, thresholds):
    return {cls: conf for cls, conf in predictions.items() if conf >= thresholds.get(cls, 0)}

# Function to load models and cache them
@st.cache_resource
def load_model_cached(model_path):
    return load_model(model_path)

# Load the grape and pomegranate models
grape_model_path = 'grape_and_Pomogranate_disease_2.0.h5'
pomegranate_model_path = 'grape_and_Pomogranate_disease_2.0.h5'

try:
    grape_model = load_model_cached(grape_model_path)
    pomegranate_model = load_model_cached(pomegranate_model_path)
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Apply custom CSS for background, image, and prediction box styling
st.markdown("""
    <style>
    /* Apply gradient background with an overlay image */
    .reportview-container, .sidebar .sidebar-content {
        background: url('https://www.transparenttextures.com/patterns/asfalt-dark.png'), linear-gradient(to bottom, #3b0a45, #000000);
        color: #ffffff;
    }
    /* Reduce image size with rounded corners and shadow */
    .stImage img {
        max-width: 40%; /* Reduced size */
        border-radius: 20px;
        border: 3px solid #6a1b9a;
        margin: 0 auto;
        display: block;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
        transition: box-shadow 0.3s;
    }
    .stImage img:hover {
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.7);
    }
    /* Style for the prediction box with gradient background and shadow */
    .prediction-box {
        border: 2px solid #6a1b9a;
        border-radius: 10px;
        padding: 15px;
        background: linear-gradient(to bottom, #6a1b9a, #000000);
        color: white;
        text-align: center;
        font-size: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .prediction-box:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.7);
    }
    h1 {
        font-family: 'Arial', sans-serif;
        font-size: 2.5rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 20px;
    }
    p, .stMarkdown {
        font-family: 'Arial', sans-serif;
        font-size: 1.2rem;
        color: #ffffff;
        text-align: center;
    }
    /* Add subtle animation to icons or any additional features */
    .feature-icon {
        font-size: 50px;
        color: #ffffff;
        margin: 10px;
        transition: transform 0.3s, color 0.3s;
    }
    .feature-icon:hover {
        transform: scale(1.2);
        color: #6a1b9a;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app
st.title("Grape and Pomegranate Disease Prediction")
st.write("Select the type of plant and upload an image to predict the disease.")

# Add background feature icons
st.markdown('<div style="text-align: center;">'
            '<span class="feature-icon">🍇</span>'
            '<span class="feature-icon">🍈</span>'
           
            '</div>', unsafe_allow_html=True)

# Select the model
model_choice = st.selectbox("Choose a plant type:", ["Grape", "Pomegranate"])

if model_choice == "Grape":
    model = grape_model
    categories = grape_categories
    thresholds = grape_thresholds
else:
    model = pomegranate_model
    categories = pomegranate_categories
    thresholds = pomegranate_thresholds

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display loading spinner
        with st.spinner("Processing image..."):
            # Preprocess the image
            image = Image.open(uploaded_file).convert('RGB')
            image = image.resize((256, 256))
            img_array = np.array(image, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

            # Make prediction
            try:
                raw_predictions = model.predict(img_array)[0]
                st.write("Raw predictions:", raw_predictions)

                # Map raw predictions to categories
                pred_dict = dict(zip(categories, raw_predictions))

                # Apply thresholds
                filtered_predictions = apply_thresholds(pred_dict, thresholds)
                st.write("Filtered predictions after applying thresholds:", filtered_predictions)

                # Display top prediction
                if filtered_predictions:
                    top_prediction = max(filtered_predictions, key=filtered_predictions.get)
                    confidence = filtered_predictions[top_prediction]
                    st.markdown(f"""
                        <div class="prediction-box">
                            <b>Prediction:</b> {top_prediction} <br>
                            <b>Confidence:</b> {confidence:.2f}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display disease information
                    st.markdown(f"""
                        <div class="prediction-box">
                            <b>About {top_prediction}:</b> {disease_info.get(top_prediction, "No information available.")}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.write("No valid predictions after applying thresholds.")

            except Exception as e:
                st.error(f"Error during prediction: {e}")

    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.stop()
