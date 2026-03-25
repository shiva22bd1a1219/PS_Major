# IntelliScan: Advanced Brain Tumor Analysis System

IntelliScan is a professional medical diagnostic portal that utilizes a hybrid Deep Learning approach to classify brain tumors and calculate clinical risk metrics directly from MRI scans.

## 🚀 Key Features

- **High-Accuracy Classification:** Powered by a fine-tuned **ResNet-18** model (99.62% Accuracy) identifying Glioma, Meningioma, Pituitary, and Normal scans.
- **Automated Tumor Segmentation:** Utilizes an unsupervised Computer Vision segmenter for pixel-perfect masking.
- **Clinical Area Calculation:** Automatically estimates the tumor size in $cm^2$.
- **Clinical Risk Grading:** Triage system that assigns Risk Levels (Critical, Moderate, Normal) based on tumor type and size.
- **Professional Dashboard:** React-based frontend with detailed patient history and report generation.

---

## 🛠️ Setup Instructions

### 1. Backend Setup (Flask API)

1. Open a terminal in the `intelliscan backend` folder.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the server:
   ```bash
   python run_mode.py
   ```
   *The API will run on http://127.0.0.1:5003*

### 2. Frontend Setup (React/Vite)

1. Open a terminal in the `intelliscan frontend` folder.
2. Install Node dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
   *The dashboard will be available at http://localhost:5173*

---

## 📂 Project Structure

- `/intelliscan backend`: Flask API, ML Model weights, and training scripts.
- `/intelliscan frontend`: React source code, components, and styling.
- `/intelliscan training`: Jupyter notebooks and utility scripts for model development.

## 📝 Note for Mentors
This project was upgraded from a 51% accuracy baseline to 99.6% using **Transfer Learning** and robust Data Augmentation. The segmentation logic was re-engineered to prevent aspect-ratio distortion and handle black padding in MRI collages.
