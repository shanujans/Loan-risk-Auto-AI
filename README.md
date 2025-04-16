# Loan Risk Prediction using IBM AutoAI & Snap Boosting

![Machine Learning](https://img.shields.io/badge/-Machine%20Learning-blueviolet) 
![IBM Cloud](https://img.shields.io/badge/IBM-Cloud-blue) 
![AutoAI](https://img.shields.io/badge/AutoAI-FF6D00?logo=ibm) 
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E)

This project uses IBM Watson Studio's AutoAI to build a machine learning pipeline that predicts loan applicant risk ("Good" or "Bad"). The Snap Boosting Machine Classifier was selected as the best performing model through AutoAI's automated evaluation process.

# Key Results

- **Best Model**: Snap Boosting Machine Classifier
- **Evaluation Metric**: Accuracy
- **Test Score**: [Insert actual accuracy score from notebook here]
- **Positive Label**: "No Risk"
- **Random State**: 33 (for reproducibility)

# Technical Implementation

### Pipeline Architecture
The AutoAI-generated pipeline includes:

1. **Data Preprocessing**:
   - Column selection and type conversion
   - Missing value handling (imputation)
   - Categorical encoding (ordinal)
   - Feature scaling (optional)

2. **Feature Engineering**:
   - Feature union for numerical and categorical features
   - Feature transformations (PCA)
   - Automated feature selection

3. **Modeling**:
   - Snap Boosting Machine Classifier with parameters:
     - Learning rate: 0.295
     - Max depth: 3
     - Number of rounds: 83
     - Class weight: balanced

### Key Parameters
```python
snap_boosting_machine_classifier = SnapBoostingMachineClassifier(
    class_weight="balanced",
    gpu_ids=[0],
    learning_rate=0.29527603863901886,
    max_max_depth=3,
    min_max_depth=3,
    num_round=83,
    random_state=33
)
