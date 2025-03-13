# Predicting-Diabetes-Risk-Using-Health-Parameters-A-Machine-Learning-Approach
 Project Overview This project aims to develop a machine learning model that predicts diabetes risk based on key health parameters. By analyzing factors like age, BMI, glucose levels, blood pressure, and family history, this model provides a data-driven approach to early diabetes detection, aiding in preventive healthcare decisions.
# 📊 Dataset
The dataset consists of medical records containing various health indicators that influence diabetes risk. Key features include:
* Glucose Level
* Blood Pressure
* BMI (Body Mass Index)
* Age & Gender
* Family History of Diabetes
  
# Physical Activity Levels
🏗 Machine Learning Approach
For this classification task, the K-Nearest Neighbors (KNN) algorithm was used. The model was fine-tuned with cross-validation (CV) to improve performance and reduce overfitting.

# 📈 Model Performance & Results
The model was evaluated using cross-validation to ensure robustness.

✅ Training Accuracy after CV: 73.47%
✅ Test Accuracy after CV: 70.4%

# Performance was assessed using:

* Accuracy Score
* Confusion Matrix
* Precision, Recall, and F1-Score
* 
# 🛠 Tools & Technologies Used
🚀 Programming Language: Python
📊 Libraries: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
🖥 Model Evaluation: Cross-Validation & Confusion Matrix

# 📌 Key Findings
🔹 The KNN model performed with a test accuracy of 70.4% after cross-validation.
🔹 Glucose level, BMI, and age emerged as the most significant factors affecting diabetes risk.
🔹 The model's accuracy could be improved with feature scaling, hyperparameter tuning, or a different ML algorithm.

# 🚀 Future Enhancements
🔹 Experimenting with other models (Random Forest, XGBoost, Deep Learning)
🔹 Feature engineering to improve prediction accuracy
🔹 Deploying the model using Flask or Streamlit for real-time predictions

