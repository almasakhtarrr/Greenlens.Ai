import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("🌍 AI Waste Intelligence & Carbon Impact System")

# Load Pretrained Model
from tensorflow.keras.models import load_model
model = load_model("waste_classifier.h5")
# Upload Image
uploaded_file = st.file_uploader("Upload Real-World Waste Image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Input Image", use_column_width=True)

    # Preprocessing
    img_resized = img.resize((224,224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # AI Detection
    preds = model.predict(img_array)

    class_id = preds[0].argmax()
    confidence = preds[0][class_id] * 100

    predicted_class = class_names[class_id]

    st.subheader("🧠 Detected Waste Type")
    st.write(f"{predicted_class} - {confidence:.2f}%")

    # Waste Mapping
    recyclable = ["bottle","can","carton"]
    biodegradable = ["banana","apple","orange"]
    hazardous = ["battery","lighter"]

    total_carbon_landfill = 0
    total_carbon_saved = 0

    st.subheader("♻ Waste Category & Carbon Impact")

    for obj in detected_items:

        if any(word in obj for word in recyclable):
            category = "Recyclable"
            landfill = 3
            saved = 2
        elif any(word in obj for word in biodegradable):
            category = "Biodegradable"
            landfill = 4
            saved = 3
        elif any(word in obj for word in hazardous):
            category = "Hazardous"
            landfill = 8
            saved = 5
        else:
            category = "General"
            landfill = 5
            saved = 2

        total_carbon_landfill += landfill
        total_carbon_saved += saved

        st.write(f"{obj} → {category}")
        st.write(f"Landfill CO₂: {landfill} kg | Proper Segregation Saves: {saved} kg")
        st.write("---")

    # Carbon Comparison
    st.subheader("🔥 Carbon Consequence Simulator")

    col1, col2 = st.columns(2)

    with col1:
        st.error(f"If Landfilled: {total_carbon_landfill} kg CO₂")

    with col2:
        st.success(f"If Properly Segregated: {total_carbon_saved} kg CO₂ Saved")

    # Impact Meter
    st.subheader("🌱 Carbon Impact Level")

    if total_carbon_landfill < 5:
        st.success("Low Impact")
    elif total_carbon_landfill < 12:
        st.warning("Moderate Impact")
    else:
        st.error("High Impact")

    # Green Score
    green_score = total_carbon_saved * 10
    st.subheader("🏆 Green Impact Score")
    st.write(f"Green Credits Earned: {green_score}")

    # Dashboard Chart
    st.subheader("📊 Waste Carbon Distribution")

    plt.figure()
    plt.bar(["Landfill","Saved"], [total_carbon_landfill,total_carbon_saved])
    st.pyplot(plt)

    # Predictive Insight
    st.subheader("📈 Predictive Carbon Trend")

    months = np.array(range(1,7)).reshape(-1,1)
    carbon_history = np.array([5,7,6,8,9,10])
    model_lr = LinearRegression()
    model_lr.fit(months, carbon_history)

    future = np.array([[7],[8],[9]])
    prediction = model_lr.predict(future)

    plt.figure()
    plt.plot(months, carbon_history)
    plt.plot(future, prediction)
    plt.xlabel("Month")
    plt.ylabel("Carbon Impact")
    st.pyplot(plt)

    st.info("If current trend continues, carbon impact may increase in coming months.")
