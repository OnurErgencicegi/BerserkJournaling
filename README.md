# 🧠 Berserk Journaling: AI-Powered Emotion Diary

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-green)

**Berserk Journaling** is an intelligent journaling application that uses Deep Learning to analyze your daily entries and extract the underlying emotional tone. Unlike standard keyword-based tools, it uses a custom-trained neural network to understand context, helping users track their mental well-being over time.

---

## 🚀 Features

* **Real-time Emotion Analysis:** Instantly detects 6 core emotions: *Sadness, Joy, Love, Anger, Fear, Surprise*.
* **Custom Deep Learning Model:** Built from scratch using **PyTorch**, employing a **CNN (Convolutional Neural Network)** architecture for efficient text classification.
* **Journaling Interface:** A clean, modern UI built with Streamlit to write and save daily thoughts.
* **History & Tracking:** Automatically saves entries and visualizes emotional trends via a sidebar history.
* **Smart Preprocessing:** Includes custom text cleaning, tokenization, and dynamic vocabulary building.

---

## 🏗️ Model Architecture (Built from Scratch)

This project does **not** rely on pre-trained APIs for the core logic. The neural network was designed and trained specifically for this application:

* **Embedding Layer:** Converts tokenized text into dense vector representations.
* **Convolutional Layers (Conv1d):** Three parallel convolutional layers with different kernel sizes (3, 4, 5) act as feature extractors, capturing N-gram patterns (phrases) indicative of specific emotions.
* **Max Pooling:** Extracts the most salient features from the convolutional maps.
* **Class Balancing:** Implements **Weighted Cross Entropy Loss** to handle dataset imbalance (e.g., distinguishing rare "Surprise" examples from frequent "Joy" examples).
* **Regularization:** Uses **Dropout** layers to prevent overfitting on the training data.

---

## 📂 Project Structure

The project follows a modular "Separation of Concerns" principle:

```text
berserk_journaling/
│
├── core/                   # Backend Logic & AI Core
│   ├── config.py           # Hyperparameters (Epochs, LR, Dimensions)
│   ├── model.py            # Custom PyTorch CNN Architecture
│   ├── data_loader.py      # Data fetching & Preprocessing pipeline
│   ├── predictor.py        # Inference engine for the UI
│   └── utils.py            # Helper functions (Text cleaning, Vocab building)
│
├── models/                 # Saved model weights (.pth files)
├── app.py                  # Frontend Application (Streamlit)
├── train.py                # Training Script
└── requirements.txt        # Dependencies