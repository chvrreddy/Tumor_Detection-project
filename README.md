# 🧬 Tumor Detection Project

## 📌 Project Overview  
This project builds a machine learning pipeline to **detect tumors** as **Benign (0)** or **Malignant (1)** based on medical imaging features.  
The dataset used is **Tumor_Detection.csv** containing 569 samples with 30 numeric features.  

We trained a **Random Forest Classifier**, performed **EDA**, evaluated performance, and saved the model for future predictions.  

---

## 📂 Dataset  
- **File:** `Tumor_Detection.csv`  
- **Shape:** 569 rows × 32 columns  
- **Target column:** `diagnosis` (`B = 0, Benign` and `M = 1, Malignant`)  
- **Features:** 30 numeric attributes (e.g., radius_mean, texture_mean, area_worst, etc.)  
- **Dropped column:** `id` (not useful for prediction)  

---

## ⚙️ Steps Performed
1. **Data Cleaning**
   - Dropped `id` and unnamed columns  
   - Encoded `diagnosis` → `B=0`, `M=1`  

2. **Exploratory Data Analysis (EDA)**
   - Class distribution plot (Benign vs Malignant)  
   - Histograms and correlation heatmap  

3. **Preprocessing**
   - Feature scaling using `StandardScaler`  
   - Train-test split (80:20 with stratification)  

4. **Model Training**
   - Random Forest Classifier (`n_estimators=200`)  

5. **Evaluation**
   - Accuracy, ROC AUC, confusion matrix, classification report  
   - 5-fold Cross-validation  
   - Feature importance ranking  

6. **Model Saving**
   - Trained model saved as `tumor_rf_model.pkl`  
   - Scaler saved as `scaler.pkl`  

7. **Prediction**
   - Example code included for predicting new samples  

---

## 📊 Results
- **Test Accuracy:** ~96%  
- **ROC AUC:** ~0.99  
- **5-fold CV Accuracy:** ~95%  
- **Top features:**  
  - `perimeter_worst`  
  - `area_worst`  
  - `concave points_worst`  
  - `concave points_mean`  
  - `radius_worst`  

---



---

## 🚀 How to Run
1. Clone the repository or download files.  
2. Install requirements:  
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib
