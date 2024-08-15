import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Define a function to load the model and cache it
@st.cache_resource
def load_model_cached(model_path):
    return load_model(model_path)

# Load the trained model
model_path = 'grape_and_Pomogranate_disease_streamlit.h5'
try:
    model = load_model_cached(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define categories
categories = ["Black Rot", "ESCA", "Healthy", "Leaf Blight", "Healthy_Pomogranate", "Cercospora", "Bacterial_Blight", "Anthracnose"]

# Apply custom CSS for background and prediction box styling
st.markdown("""
    <style>
    /* Apply gradient background to the entire page */
    .reportview-container, .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #3b0a45, #000000) !important;
        color: #ffffff;
    }
    /* Style for the image with rounded corners and shadow */
    .stImage img {
        max-width: 60%;
        border-radius: 20px; /* Smooth, rounded corners */
        border: 3px solid #6a1b9a; /* Optional: add border color matching the theme */
        margin: 0 auto;
        display: block;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5); /* Shadow effect */
        transition: box-shadow 0.3s; /* Smooth transition */
    }
    .stImage img:hover {
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.7); /* Enhanced shadow on hover */
    }
    /* Style for the prediction box with gradient background and shadow */
    .prediction-box {
        border: 2px solid #6a1b9a;
        border-radius: 10px;
        padding: 15px;
        background: linear-gradient(to bottom, #6a1b9a, #000000); /* Dark violet to black gradient */
        color: white;
        text-align: center;
        font-size: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5); /* Shadow effect */
        transition: transform 0.3s, box-shadow 0.3s; /* Smooth transition */
    }
    .prediction-box:hover {
        transform: scale(1.05); /* Slightly enlarge on hover */
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.7); /* Enhance shadow on hover */
    }
    /* Style for the title */
    h1 {
        font-family: 'Arial', sans-serif;
        font-size: 2.5rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 20px;
    }
    /* Style for the text */
    p, .stMarkdown {
        font-family: 'Arial', sans-serif;
        font-size: 1.2rem;
        color: #ffffff;
        text-align: center;
    }
    /* Loader styles */
    .stSpinner {
        margin: 0 auto;
        display: block;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app
st.title("Grape And Pomogranate Disease Prediction")
st.write("Upload an image of a grape leaf or Pomogranate fruit to predict the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display loading spinner
        with st.spinner("Processing image..."):
            image = Image.open(uploaded_file)
            image = image.resize((256, 256))
            img_array = np.array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
            
            # Make prediction
            try:
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions[0])
                predicted_label = categories[predicted_class]
                confidence = predictions[0][predicted_class]

                # Display prediction in a styled box with bold text
                st.markdown(f"""
                    <div class="prediction-box">
                        <b>Prediction:</b> {predicted_label} <br>
                        <b>Confidence:</b> {confidence:.2f}
                    </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error during prediction: {e}")

    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.stop()