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
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Apply custom CSS for enhanced styling
st.markdown("""
    <style>
    /* Apply gradient background with animated transition */
    body {
        background: linear-gradient(120deg, #6a1b9a, #3b0a45, #000000);
        background-size: 300% 300%;
        animation: gradientAnimation 15s ease infinite;
        color: #ffffff;
        font-family: 'Roboto', sans-serif;
    }

    @keyframes gradientAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Stylish image display with rotation effect */
    .stImage img {
        max-width: 60%;
        border-radius: 30px;
        border: 5px solid #ffffff;
        margin: 0 auto;
        display: block;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.5);
        transition: transform 0.5s ease, box-shadow 0.5s ease;
    }
    .stImage img:hover {
        transform: rotate(3deg) scale(1.05);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.7);
    }

    /* Stylish and interactive buttons */
    .stButton button {
        border-radius: 30px;
        padding: 0.8rem 1.6rem;
        font-size: 1.2rem;
        font-weight: bold;
        color: #ffffff;
        background: linear-gradient(45deg, #6a1b9a, #3b0a45);
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background: linear-gradient(45deg, #3b0a45, #6a1b9a);
        transform: scale(1.1);
    }

    /* Gradient and shadow for prediction box */
    .prediction-box {
        border: 2px solid #ffffff;
        border-radius: 15px;
        padding: 20px;
        background: linear-gradient(45deg, #6a1b9a, #3b0a45);
        color: white;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 30px;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .prediction-box:hover {
        transform: scale(1.05);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.7);
    }

    /* Enhanced title styling */
    h1 {
        font-family: 'Roboto', sans-serif;
        font-size: 3rem;
        font-weight: 900;
        color: #ffffff;
        text-align: center;
        margin-bottom: 30px;
    }

    /* General text styling */
    p, .stMarkdown {
        font-family: 'Roboto', sans-serif;
        font-size: 1.4rem;
        color: #ffffff;
        text-align: center;
    }

    </style>
""", unsafe_allow_html=True)

# Streamlit app
st.title("Grape and Pomegranate Disease Prediction")
st.write("Select the type of plant and upload an image to predict the disease.")

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
                else:
                    st.write("No valid predictions after applying thresholds.")

            except Exception as e:
                st.error(f"Error during prediction: {e}")

    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.stop()
