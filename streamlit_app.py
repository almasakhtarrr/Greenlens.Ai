import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from ultralytics import YOLO

st.set_page_config(layout="wide")
st.title("🌍 AI Waste Intelligence & Carbon Impact System")

# Load YOLO model
model = YOLO("yolov8n.pt")   # use your trained model later

uploaded_file = st.file_uploader(
    "Upload Real-World Waste Image", type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    img = Image.open(uploaded_file).convert("RGB")

    # Run YOLO
    results = model(np.array(img))

    result = results[0]

    # Draw detections
    annotated = result.plot()
    st.image(annotated, caption="Detected Waste Objects",
             use_container_width=True)

    # ----------------------------------
    # Extract detected class names
    # ----------------------------------
    detected_items = []

    names = result.names
    if result.boxes is not None:
        for c in result.boxes.cls:
            detected_items.append(names[int(c)])

    if len(detected_items) == 0:
        st.warning("No objects detected.")
        st.stop()

    st.subheader("🧠 Detected Objects")
    st.write(detected_items)

    # ----------------------------------
    # Map object → waste category
    # ----------------------------------

    recyclable = ["bottle", "cup", "can", "plastic bottle", "carton"]
    biodegradable = ["banana", "apple", "orange", "food", "sandwich"]
    hazardous = ["battery", "cell phone", "knife", "scissors"]

    total_carbon_landfill = 0
    total_carbon_saved = 0

    st.subheader("♻ Waste Category & Carbon Impact")

    for obj in detected_items:

        name = obj.lower()

        if any(x in name for x in recyclable):
            category = "Recyclable"
            landfill = 3
            saved = 2

        elif any(x in name for x in biodegradable):
            category = "Biodegradable"
            landfill = 4
            saved = 3

        elif any(x in name for x in hazardous):
            category = "Hazardous"
            landfill = 8
            saved = 5

        else:
            category = "General"
            landfill = 5
            saved = 2

        total_carbon_landfill += landfill
        total_carbon_saved += saved

        st.write(f"**{obj} → {category}**")
        st.write(f"Landfill CO₂: {landfill} kg | Saved: {saved} kg")
        st.write("---")

    # ----------------------------------
    # Carbon comparison
    # ----------------------------------

    st.subheader("🔥 Carbon Consequence Simulator")

    col1, col2 = st.columns(2)

    with col1:
        st.error(f"If Landfilled: {total_carbon_landfill} kg CO₂")

    with col2:
        st.success(f"If Properly Segregated: {total_carbon_saved} kg CO₂ Saved")

    # ----------------------------------
    # Impact meter
    # ----------------------------------

    st.subheader("🌱 Carbon Impact Level")

    if total_carbon_landfill < 5:
        st.success("Low Impact")
    elif total_carbon_landfill < 12:
        st.warning("Moderate Impact")
    else:
        st.error("High Impact")

    # ----------------------------------
    # Green score
    # ----------------------------------

    green_score = total_carbon_saved * 10
    st.subheader("🏆 Green Impact Score")
    st.write(f"Green Credits Earned: {green_score}")

    # ----------------------------------
    # Dashboard chart
    # ----------------------------------

    st.subheader("📊 Waste Carbon Distribution")

    fig = plt.figure()
    plt.bar(["Landfill", "Saved"],
            [total_carbon_landfill, total_carbon_saved])
    st.pyplot(fig)
    plt.close()

    # ----------------------------------
    # Predictive insight
    # ----------------------------------

    st.subheader("📈 Predictive Carbon Trend")

    months = np.array(range(1, 7)).reshape(-1, 1)
    carbon_history = np.array([5, 7, 6, 8, 9, 10])

    model_lr = LinearRegression()
    model_lr.fit(months, carbon_history)

    future = np.array([[7], [8], [9]])
    prediction = model_lr.predict(future)

    fig2 = plt.figure()
    plt.plot(months, carbon_history, label="Past")
    plt.plot(future, prediction, label="Predicted")
    plt.xlabel("Month")
    plt.ylabel("Carbon Impact")
    plt.legend()
    st.pyplot(fig2)
    plt.close()

    st.info(
        "If current waste trend continues, carbon impact may increase in coming months."
    )
