## END TO END MACHINE LEARNING PROJECT

# 🎓 Student Math Score Predictor
**An End-to-End Machine Learning Solution for Academic Performance Analysis**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://student-performance-s20230030395.streamlit.app/)

## 🚀 Live Demo
Access the live application here: [Student Math Score Predictor](https://student-performance-s20230030395.streamlit.app/)

## 📌 Project Overview
This project predicts a student's **Math Score** based on demographic data and scores in other subjects. Built with **Modular Coding** standards, this project demonstrates a production-ready MLOps pipeline—moving away from monolithic Jupyter notebooks to a structured, scalable Python package.

## 🧠 Model Evaluation & Selection
During the development phase, I implemented a comprehensive model training pipeline that evaluated the following algorithms:

* **Linear Regression** (Baseline)
* **Decision Tree** & **Random Forest** (Tree-based Ensembles)
* **Gradient Boosting**, **AdaBoost**, **XGBoost**, & **CatBoost** (Boosting techniques)
* **K-Nearest Neighbors (KNN)** (Instance-based learning)

### 📊 Performance Comparison
| Algorithm | R2 Score | Status |
| :--- | :--- | :--- |
| **Linear Regression** | **~0.88** | **Selected Model** |
| **CatBoost** | ~0.87 | Evaluated |
| **Gradient Boosting** | ~0.86 | Evaluated |
| **Random Forest** | ~0.85 | Evaluated |
| **XGBoost** | ~0.84 | Evaluated |
| **KNN** | ~0.78 | Evaluated |



### ❓ Why did Linear Regression outperform Boosting/Tree models?
In many Machine Learning projects, there is a tendency to assume "complex is better." However, in this specific dataset, **Linear Regression** achieved the highest accuracy for three key reasons:

1. **Strong Multicollinearity:** The independent variables (specifically Reading and Writing scores) have a near-perfect linear relationship with the target (Math score). Linear models excel at capturing these direct relationships.
2. **Data Sparsity:** With a relatively small dataset, complex boosting models like XGBoost or CatBoost can sometimes "overfit" the noise, whereas Linear Regression remains robust.
3. **Simplicity & Generalization:** The "Law of Parsimony" applies here; a simpler model generalizes better when the underlying data patterns are naturally linear.



## 🏗️ Project Architecture
This project follows professional software engineering practices:
* `src/components`: **Data Ingestion**, **Data Transformation**, and **Model Trainer**.
* `src/pipeline`: **Predict Pipeline** for real-time inference.
* `src/logger.py` & `src/exception.py`: Custom logging and robust error handling.
* `artifacts/`: Persistent storage for the trained model and preprocessor.



## 🏁 Installation & Local Setup
1. **Clone the Repo:**
   ```bash
   git clone [https://github.com/sivamani-muraboyina/mlproject.git](https://github.com/sivamani-muraboyina/mlproject.git)
   cd mlproject

2. **Environment Setup**
   ```bash
   pip install -r requirements.txt

3. **to predict using Linear Regressor**
   ```bash
   streamlit run streamlit_app.py


## Developed by: Sivamani Muraboyina