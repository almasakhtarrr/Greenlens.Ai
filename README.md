# GreenLens.Ai 🌱  
AI-Based Waste Segregation & Carbon Footprint Analyzer

It is a computer vision powered sustainability platform designed for
real-world environments. The system detects multiple waste objects from cluttered,
non-studio images and converts each detection into measurable carbon impact insights
through a real-time analytics dashboard.

The project focuses on practical deployment for campuses, institutions and smart
waste management systems.

---

# Key Features

- Real-world waste image support (no white background assumption)
- Multi-object waste detection using YOLO-style object detection
- Automatic classification into:
  - Biodegradable
  - Recyclable
  - Hazardous
- Carbon consequence comparison:
  - Landfilled vs Properly Segregated
- Real-time carbon impact meter
- Green impact score and credit system
- Campus sustainability analytics dashboard
- Predictive carbon footprint insights

---

# Final Project Workflow

### 1. Real-World Image Input
Users upload an image or capture a live photo.  
Images may contain multiple objects, cluttered backgrounds, poor lighting and
overlapping waste items.

The system is designed for real-life conditions.

---

### 2. Preprocessing Layer
- Image resizing and normalization
- Lighting and contrast normalization
- Optional lightweight augmentation

Purpose: improve model robustness and generalization.

---

### 3. AI Waste Detection (Object Detection Model)

A YOLO-style object detection model is used to:

- Detect multiple waste objects in a single image
- Draw bounding boxes
- Classify each detected object into:
  - Biodegradable
  - Recyclable
  - Hazardous

Each detection is displayed with confidence score  
(e.g. “Plastic Bottle – 89%”).

---

### 4. Carbon Consequence Simulator (Core Innovation)

For every detected waste object, the system compares two scenarios:

#### If Landfilled
- Estimated CO₂ equivalent generated
- Methane emission impact
- Approximate decomposition time

#### If Properly Segregated
- Estimated CO₂ saved
- Energy saved
- Resource recovery benefit

A split-view comparison is generated to show environmental consequences clearly.

---

### 5. Real-Time Carbon Impact Meter

A dynamic impact gauge is calculated using the total detected waste impact:

- 🌱 Low impact
- ⚠ Moderate impact
- 🔥 High impact

This provides an instant sustainability indicator for the uploaded image.

---

### 6. Green Impact Score System

- Green credits are assigned for proper segregation
- User score is updated
- Campus / organization cumulative score is maintained

This introduces a gamification layer to encourage responsible behavior.

---

### 7. Campus Sustainability Dashboard

The dashboard displays:

- Total CO₂ saved (weekly / monthly)
- Waste category distribution
- Carbon impact trend graph
- Top contributors leaderboard

---

### 8. Predictive Insight Layer

A simple trend-based prediction module estimates future carbon footprint.

Example:
“If the current waste trend continues, projected carbon footprint will increase
by X% in the next 30 days.”

This demonstrates scalability and future readiness.

---

## 🏗️ Simplified YOLO Deployment Strategy

To ensure fast training, stable inference and reliable demo:

- A lightweight YOLO model variant is used
- Only three target classes are trained
- Limited custom dataset is fine-tuned on top of a pre-trained model
- Single-stage detection + classification is performed

This avoids heavy multi-model pipelines and improves real-time performance.

---

## 🗂️ Project Structure

GreenLens.Ai/
