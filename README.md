# 🧠 Fraud Detection using LSTM on IBM LinuxONE
**IBM Z Datathon 2026 — Machine Learning Research**
Predictive AI: Credit Card Fraud Detection using 2-Layer LSTM on IBM LinuxONE.

[![Main Project](https://img.shields.io/badge/Main_Project-Auditor's_AI-052FAD?style=for-the-badge&logo=ibm&logoColor=white)](https://github.com/kaysanchezzzz/IBM-Z-Datathon-Auditor-AI)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)]()
[![IBM LinuxONE](https://img.shields.io/badge/IBM_LinuxONE-s390x-052FAD?style=for-the-badge&logo=ibm&logoColor=white)]()

> **Author:** Syakir Kauthar Bin Sahril (Bachelor of Accounting Hons. Student)  
> **Objective:** To build a high-recall anomaly detection model for forensic accounting, running natively on enterprise mainframe architecture.

---

### 📖 Project Overview
This repository contains the raw Machine Learning pipeline for **"The Auditor's AI"**. 

I trained a 2-layer Long Short-Term Memory (LSTM) neural network using TensorFlow on an **IBM LinuxONE (s390x)** mainframe. The model analyzes sequences of 7 historical transactions to detect anomalies, achieving a **99.86% accuracy** and a **100% Recall rate**.

---

### 🏗️ The ML Pipeline
1. **Data Ingestion:** Loaded 245,864 historical credit card transactions.
2. **Feature Engineering:** Custom encoders for time, amount (log-scaled), and merchant locations. 
3. **Sequence Windowing:** Grouped transactions into 7-step sequences per user/card.
4. **LSTM Training:** Utilized IBM's `NNPA` (Neural Network Processing Assist) hardware acceleration on the mainframe.
5. **Export:** Saved the trained weights and exported the model to **ONNX** format for production integration.

---

### 📂 Repository Contents
*   `Fraud_LSTM_Keras_TF (LEGIT).ipynb` - The complete Jupyter Notebook containing data preprocessing, model definition, training loop, and ONNX export.

---
*Created for the IBM Z Datathon 2026.*
