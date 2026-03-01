import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils.processor import run_inference, draw_boxes
from utils.carbon_logic import get_carbon_metrics

st.set_page_config(page_title="Campus EcoTrack AI", layout="wide")

st.title("🌍 Campus EcoTrack AI")
st.markdown("### Real-Time Waste Analysis & Carbon Consequences")

# Load Class Names
with open("models/classes.txt", "r") as f:
    CLASSES = [line.strip() for line in f.readlines()]

# Sidebar: Gamification
st.sidebar.header("🏆 Green Leaderboard")
st.sidebar.metric("Campus Credits", "14,250", "+128 today")

uploaded_file = st.file_uploader("Upload an image of waste...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. Image Preprocessing
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # 2. AI Inference
    detections = run_inference(image)
    annotated_img = draw_boxes(image.copy(), detections, CLASSES)
    
    # 3. Split-Screen Layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 AI Detection View")
        st.image(annotated_img, channels="BGR", use_container_width=True)
        
    with col2:
        st.subheader("🔥 Carbon Impact Simulator")
        total_co2_saved = 0
        
        if not detections:
            st.write("No waste detected. Try a clearer image!")
        
        for det in detections:
            label = CLASSES[det['class_id']]
            impact = get_carbon_metrics(label)
            total_co2_saved += impact['co2_saved']
            
            with st.expander(f"📦 Detected: {label}"):
                c1, c2 = st.columns(2)
                c1.error(f"Landfill: {impact['co2_landfill']}kg CO2")
                c2.success(f"Recycled: {impact['co2_recycled']}kg CO2")
                st.write(f"**Action:** {impact['info']}")

    # 4. Impact Meter
    st.divider()
    st.subheader("🌱 Session Impact Meter")
    st.progress(min(total_co2_saved / 5.0, 1.0)) # Scales to 5kg of CO2 saved
    st.write(f"This session potentially saved **{total_co2_saved:.2f} kg of CO2 emissions**!")

    # --- Dataset Transparency Section ---
    with st.expander("📊 View Model Training Samples"):
    st.write("These are real-world 'messy' images used to train this AI.")
    # Show samples side-by-side
    col_a, col_b = st.columns(2)
    col_a.image("data_sample/sample_1.jpg", caption="Cluttered Floor Sample")
    col_b.image("data_sample/sample_2.jpg", caption="Overlapping Waste Sample")
