# breast-cancer-detection-using-naive-bayes
Breast Cancer Classification using Gaussian Naive Bayes

This script demonstrates how to classify breast cancer tumors as
malignant or benign using the Gaussian Naive Bayes algorithm from
scikit-learn.

Steps:
1. Load the Breast Cancer dataset.
2. Split the dataset into training and testing sets.
3. Train a Gaussian Naive Bayes model.
4. Make predictions on the test set.
5. Evaluate model accuracy.


Requirements:

pip install scikit-learn numpy


How to Run:

python breast_cancer_nb.py

Code Explanation
 load_breast_cancer(): Loads the dataset.
  train_test_split(): Divides data into training and testing sets.
  GaussianNB(): Initializes the Naive Bayes classifier.
  fit(): Trains the model.
  predict(): Makes predictions.
  accuracy_score(): Measures accuracy of predictions.

Expected Output:

- Class labels (malignant / benign)
- Example feature values
- Model predictions
- Model accuracy score
