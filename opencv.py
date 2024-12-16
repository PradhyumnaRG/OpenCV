import cv2
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors

def analyze_dominant_colors(image, num_colors=10):
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, random_state=42).fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    percentages = np.bincount(kmeans.labels_) / len(kmeans.labels_) * 100
    hex_colors = [mcolors.to_hex(color / 255.0) for color in colors]
    
    return pd.DataFrame({
        'Color': [f'<div style="background-color:{hex};width:30px;height:30px;'
                  f'border-radius:5px;display:inline-block;"></div>' for hex in hex_colors],
        'Hex': hex_colors,
        'RGB': [tuple(color) for color in colors],
        'Percentage': percentages
    }).sort_values(by='Percentage', ascending=False).reset_index(drop=True)

def main():
    st.set_page_config(page_title="Image Color Analysis", layout="wide", page_icon="🎨")
    
    st.markdown("""
        <style>
        body {
            background: linear-gradient(135deg, #f4f4f9, #e0e7ff);
            color: #2c3e50;
            font-family: sans-serif;
        }
        .center-content {
            text-align: center;
            margin-bottom: 20px;
        }
        .stImage img {
            margin: auto;
        }
        .color-table {
            margin: auto;
            display: table;
            text-align: center;
        }
        .color-table table {
            margin: 0 auto;
            width: 100%;  /* Increased table width */
        }
        .color-table th, .color-table td {
            padding: 15px;  /* Increased padding for more space */
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="center-content">
            <h1>🎨 Image Color Analysis</h1>
            <p>Upload an image to analyze its dominant colors and distribution.</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("Error: Unable to process the uploaded image.")
            return
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        st.write("### Image Channels")
        col1, col2, col3 = st.columns(3)
        col1.image(image[:, :, 0], caption="Red Channel", use_container_width=True)
        col2.image(image[:, :, 1], caption="Green Channel", use_container_width=True)
        col3.image(image[:, :, 2], caption="Blue Channel", use_container_width=True)
        
        with st.spinner("Analyzing colors..."):
            color_data = analyze_dominant_colors(image)
        
        st.markdown("""
            <div class="center-content">
                <h2>Color Analysis</h2>
                <div class="color-table">
                    {}
                </div>
            </div>
        """.format(color_data.to_html(escape=False, index=False)), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
