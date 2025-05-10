# Arnold Schwarzenegger Recognition with Machine Learning

##  Project Description
This project aims to enhance the security of high-profile individuals by distinguishing **Arnold Schwarzenegger** from other people using facial feature data. The task is framed as a binary classification problem using classical machine learning techniques.

The goal was to build machine learning pipelines for multiple classification algorithms and identify the best-performing model using cross-validation. A minimum test accuracy of **80%** was required to consider the model successful.

---

##  Dataset
- **Source:** `lfw_arnie_nonarnie.csv`
- **Description:** The dataset includes numerical features extracted from facial images. Each instance is labeled as either "Arnie" (Arnold Schwarzenegger) or "Non-Arnie".
- **Target Variable:** `Label` (Binary: 1 = Arnie, 0 = Non-Arnie)

---

##  Technologies Used
- Python
- pandas, numpy
- scikit-learn
- seaborn, matplotlib

---

##  Machine Learning Workflow

### 1. Data Preprocessing
- Split the data into train and test sets using stratified sampling.
- Applied standard scaling using `StandardScaler` within pipelines to normalize the features.

### 2. Models Implemented
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree Classifier**

Each model was integrated into a `Pipeline` to ensure consistent preprocessing and training.

### 3. Hyperparameter Tuning
- Used `GridSearchCV` with 5-fold `KFold` cross-validation.
- Tuned relevant hyperparameters for each model (e.g., `C`, `n_neighbors`, `max_depth`, etc.).

### 4. Model Evaluation
- Selected the best model based on cross-validation accuracy.
- Evaluated the final model on the test set using:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1 Score**
- Visualized the model’s performance using a **confusion matrix**.

---

##  Results

| Metric       | Score (Test Set) |
|--------------|------------------|
| Accuracy     | ≥ 80% ✅         |
| Precision    | ✅               |
| Recall       | ✅               |
| F1 Score     | ✅               |

The model successfully met the 80% accuracy threshold and was able to reliably classify whether a face belonged to Arnold Schwarzenegger or not.

---

