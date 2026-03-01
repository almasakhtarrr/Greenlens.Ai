import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LinearRegression

from ultralytics import YOLO


# ----------------------------------------
# Page setup
# ----------------------------------------
st.set_page_config(layout="wide")
st.title("🌍 AI Waste Intelligence & Carbon Impact System")


# ----------------------------------------
# Load YOLO model (lightweight for hackathon)
# ----------------------------------------
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

model = load_yolo()


# ----------------------------------------
# File uploader
# ----------------------------------------
uploaded_file = st.file_uploader(
    "Upload Real-World Waste Image",
    type=["jpg", "png", "jpeg"]
)


# ----------------------------------------
# Main pipeline
# ----------------------------------------
if uploaded_file:

    img = Image.open(uploaded_file).convert("RGB")

    # Run YOLO detection
    results = model(np.array(img))
    result = results[0]

    # Show detections
    annotated = result.plot()
    st.image(
        annotated,
        caption="Detected Waste Objects",
        use_container_width=True
    )

    # ----------------------------------------
    # Extract detected object names
    # ----------------------------------------
    detected_items = []

    names = result.names

    if result.boxes is not None:
        for c in result.boxes.cls:
            detected_items.append(names[int(c)])

    if len(detected_items) == 0:
        st.warning("No waste objects detected in the image.")
        st.stop()

    st.subheader("🧠 Detected Objects")
    st.write(detected_items)

    # ----------------------------------------
    # Waste category keyword mapping
    # ----------------------------------------
    recyclable = [
        "bottle", "cup", "can", "plastic", "carton", "paper"
    ]

    biodegradable = [
        "banana", "apple", "orange", "sandwich", "food"
    ]

    hazardous = [
        "battery", "cell phone", "phone", "knife", "scissors"
    ]

    total_carbon_landfill = 0
    total_carbon_saved = 0

    st.subheader("♻ Waste Category & Carbon Impact")

    # ----------------------------------------
    # Carbon consequence simulator
    # ----------------------------------------
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
        st.write(
            f"Landfill CO₂: {landfill} kg | "
            f"Proper Segregation Saves: {saved} kg"
        )
        st.write("---")

    # ----------------------------------------
    # Carbon comparison
    # ----------------------------------------
    st.subheader("🔥 Carbon Consequence Simulator")

    col1, col2 = st.columns(2)

    with col1:
        st.error(f"If Landfilled: {total_carbon_landfill} kg CO₂")

    with col2:
        st.success(
            f"If Properly Segregated: {total_carbon_saved} kg CO₂ Saved"
        )

    # ----------------------------------------
    # Impact meter
    # ----------------------------------------
    st.subheader("🌱 Carbon Impact Level")

    if total_carbon_landfill < 5:
        st.success("Low Impact")
    elif total_carbon_landfill < 12:
        st.warning("Moderate Impact")
    else:
        st.error("High Impact")

    # ----------------------------------------
    # Green score system
    # ----------------------------------------
    green_score = total_carbon_saved * 10

    st.subheader("🏆 Green Impact Score")
    st.write(f"Green Credits Earned: {green_score}")

    # ----------------------------------------
    # Dashboard (Streamlit native chart)
    # ----------------------------------------
    st.subheader("📊 Waste Carbon Distribution")

    df_bar = pd.DataFrame({
        "Type": ["Landfill", "Saved"],
        "CO₂ (kg)": [
            total_carbon_landfill,
            total_carbon_saved
        ]
    })

    st.bar_chart(df_bar.set_index("Type"))

    # ----------------------------------------
    # Predictive insight (trend-based)
    # ----------------------------------------
    st.subheader("📈 Predictive Carbon Trend")

    months = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    carbon_history = np.array([5, 7, 6, 8, 9, 10])

    lr_model = LinearRegression()
    lr_model.fit(months, carbon_history)

    future = np.array([[7], [8], [9]])
    prediction = lr_model.predict(future)

    all_months = np.concatenate(
        [months.flatten(), future.flatten()]
    )

    all_values = np.concatenate(
        [carbon_history, prediction]
    )

    df_line = pd.DataFrame({
        "Month": all_months,
        "Carbon Impact": all_values
    })

    st.line_chart(df_line.set_index("Month"))

    st.info(
        "If current waste trend continues, "
        "carbon impact is expected to rise in the coming months."
    )
