
---

# ONNX-Based Machine Learning Model  

This repository demonstrates the process of training a simple machine learning model, converting it to the ONNX format, and performing inference using the ONNX model.  

---

## 🛠 **Project Setup**  

### 🔧 **Requirements**  
Install the required libraries using:  
```bash  
pip install onnxruntime scikit-learn joblib numpy  
```  

### 📂 **Files in This Repository**  

- **`train_and_save_model.py`**:  
  Trains a machine learning model using `scikit-learn`, saves it as a `.joblib` file, and converts it to ONNX format.  
- **`load_onnx_model.py`**:  
  Loads the ONNX model and performs inference on test data.  

---

## 🚀 **How to Run the Code**  

### 1️⃣ **Train the Model and Convert it to ONNX**  
Run the following command to train the model and save it in ONNX format:  
```bash  
python train_and_save_model.py  
```  
This will generate the following files:  
- **`ml_model.joblib`**: The original machine learning model.  
- **`ml_model.onnx`**: The converted ONNX model.  

### 2️⃣ **Run Inference Using the ONNX Model**  
Run the inference code with:  
```bash  
python load_onnx_model.py  
```  
This will output predictions on the test data.  

---

## 📜 **Code Explanation**  

### **1. Training and Saving the Model (`train_and_save_model.py`)**  
- Trains a `DecisionTreeClassifier` on dummy data using `scikit-learn`.  
- Saves the trained model as `ml_model.joblib`.  
- Converts the model into ONNX format and saves it as `ml_model.onnx`.  

### **2. Loading and Running the ONNX Model (`load_onnx_model.py`)**  
- Loads the ONNX model using the `onnxruntime` library.  
- Performs inference on test data and prints the predictions.  

---

## 🌟 **What is ONNX?**  
ONNX (Open Neural Network Exchange) is an open-source format for machine learning models that allows models trained in one framework (e.g., `scikit-learn`) to be used in other frameworks with ease.  

---

---

## 📝 **License**  
This project is licensed under the MIT License.  

---

### 🎯 **Key Features**  
- Framework Portability: Convert models to ONNX format for cross-platform compatibility.  
- Efficient Inference: Leverage ONNX runtime optimizations for fast predictions.  
- Minimal Dependencies: Use only core libraries for simplicity and efficiency.  

---

