# 🏥 Maternal Health & Preeclampsia Risk Prediction System

<div align="center">

### AI-Powered Healthcare Risk Assessment Platform

Predict maternal health complications and preeclampsia risk using Machine Learning and an interactive healthcare dashboard.

</div>

---

# 📌 Overview

The **Maternal Health & Preeclampsia Risk Prediction System** is a Machine Learning-based healthcare application developed to assist in the early detection of maternal health risks and preeclampsia complications during pregnancy.

The system provides:

* Maternal health risk assessment
* Preeclampsia prediction
* AI-powered probability analysis
* Personalized medical recommendations
* Interactive healthcare interface
* Multi-step assessment workflow

This project combines:

* Machine Learning
* Healthcare analytics
* Streamlit web application development
* Frontend design using Tailwind CSS
* Responsive UI/UX principles

---

# 🎯 Problem Statement

Pregnancy complications such as **preeclampsia** are among the leading causes of maternal and fetal health risks worldwide.

Traditional diagnosis methods:

* require regular monitoring,
* depend on medical availability,
* and may delay early risk identification.

This project aims to provide an intelligent digital healthcare solution capable of:

* assisting healthcare awareness,
* identifying high-risk pregnancy indicators,
* and improving early preventive action.

---

# 🚀 Key Features

## ✅ Maternal Health Risk Prediction

Predicts overall maternal health risk based on medical and pregnancy-related parameters.

## ✅ Preeclampsia Risk Assessment

Performs a secondary assessment for preeclampsia risk when high-risk maternal conditions are detected.

## ✅ AI-Based Probability Analysis

Displays prediction probability percentages using trained Machine Learning models.

## ✅ Personalized Recommendations

Provides medical guidance and recommendations according to prediction results.

## ✅ Two-Step Workflow

Implements a smart sequential healthcare assessment process.

## ✅ Interactive User Interface

Responsive healthcare dashboard built using Streamlit and Tailwind CSS.

## ✅ Professional Landing Page

Modern animated landing page using:

* HTML
* Tailwind CSS
* Vanilla JavaScript

## ✅ Responsive Design

Optimized for:

* Desktop
* Laptop
* Tablet
* Mobile devices

---

# 🧠 Machine Learning Pipeline

The project uses supervised classification models trained on maternal healthcare datasets.

## ML Workflow

### 1️⃣ Data Collection

Healthcare-related maternal datasets were collected and preprocessed.

### 2️⃣ Data Cleaning

Performed:

* missing value handling,
* feature formatting,
* categorical conversion,
* data validation.

### 3️⃣ Feature Engineering

Prepared medical parameters for model training.

### 4️⃣ Model Training

Machine Learning classification models were trained for:

* Maternal Health Prediction
* Preeclampsia Prediction

### 5️⃣ Model Serialization

Trained models were saved using:

* Joblib (`.pkl` format)

### 6️⃣ Model Integration

Integrated directly into the Streamlit application backend.

---

# 📊 Input Parameters

## 🩺 Maternal Health Features

| Feature          | Description             |
| ---------------- | ----------------------- |
| Age              | Patient age             |
| Gravida          | Number of pregnancies   |
| Weight           | Body weight             |
| Height           | Height                  |
| Gestation Period | Pregnancy duration      |
| Blood Pressure   | Systolic & Diastolic BP |
| Anemia           | Anemia condition        |
| Albumin          | Urine albumin status    |
| Blood Sugar      | Glucose condition       |
| Fetal Position   | Baby position           |
| Fetal Heart Beat | Fetal heart rate        |
| Jaundice         | Liver condition         |
| VDRL             | Infection indicator     |
| HRsAG            | Hepatitis indicator     |

---

## 🫀 Preeclampsia Features

| Feature                | Description             |
| ---------------------- | ----------------------- |
| Age                    | Patient age             |
| Blood Pressure         | Systolic & Diastolic BP |
| BMI                    | Body Mass Index         |
| Blood Sugar            | Blood glucose level     |
| Body Temperature       | Temperature             |
| Heart Rate             | Heart beats per minute  |
| Previous Complications | Pregnancy history       |
| Diabetes History       | Diabetes indicators     |
| Mental Health          | Mental health status    |

---

# 🏗️ System Architecture

```bash
User Input
     ↓
Frontend Interface (HTML + Tailwind CSS)
     ↓
Streamlit Backend
     ↓
Machine Learning Models (.pkl)
     ↓
Prediction Engine
     ↓
Risk Probability + Recommendations
```

---

# 🛠️ Tech Stack

# Frontend

* HTML5
* Tailwind CSS
* Vanilla JavaScript

# Backend

* Python
* Streamlit

# Machine Learning

* Scikit-learn
* Pandas
* NumPy
* Joblib

# Development Tools

* VS Code
* Git
* GitHub

---

# 📂 Project Structure

```bash
preeclampsia-main/
│
├── app/
│   ├── app.py
│   ├── doctor_advice.json
│   │
│   ├── models/
│   │   ├── maternal_health_model.pkl
│   │   └── preeclampsia_model.pkl
│   │
│   └── static/
│       └── index.html
│
├── requirements.txt
├── README.md
└── .gitignore
```

---



# 🌐 Application Workflow

## Step 1 — Landing Page

Users first access the responsive healthcare landing page.

## Step 2 — Maternal Health Assessment

The system collects maternal medical information.

## Step 3 — Risk Prediction

The ML model predicts maternal health risk probability.

## Step 4 — Conditional Navigation

If high-risk conditions are detected, users proceed to preeclampsia assessment.

## Step 5 — Preeclampsia Analysis

Secondary ML model predicts preeclampsia probability.

## Step 6 — Recommendation System

The application displays:

* Risk category
* Probability score
* Medical guidance
* Recommendations

---

# 🎨 UI & UX Highlights

* Professional healthcare design
* Animated landing page
* Tailwind CSS responsive layout
* Smooth hover effects
* Scroll animations
* Interactive cards
* Mobile-friendly interface
* Modern dashboard styling

---

# ☁️ Deployment Options

This project can be deployed on:

| Platform                  | Status |
| ------------------------- | ------ |
| Streamlit Community Cloud | ✅      |
| Render                    | ✅      |
| Railway                   | ✅      |
| Hugging Face Spaces       | ✅      |

---

# 🔮 Future Improvements

Future enhancements planned for the project:

* User authentication system
* Doctor dashboard
* Database integration
* Patient history tracking
* PDF medical report generation
* Cloud-hosted ML APIs
* Real-time analytics
* Deep Learning integration
* Chatbot support

---


# 📚 Learning Outcomes

Through this project, the following concepts were implemented and learned:

* End-to-end Machine Learning workflow
* Healthcare data analysis
* Streamlit web app development
* Frontend + backend integration
* Model deployment concepts
* Interactive UI/UX design
* GitHub project management

---


