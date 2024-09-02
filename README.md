# Hospital Stay Duration and Risk Prediction Project

This project is focused on predicting the duration of hospital stay and classifying risk levels for patients using machine learning models. The models are built on a dataset containing patient information, medical history, and outcomes.

## Table of Contents

1. [Installation](#installation)
2. [Data Preprocessing](#data-preprocessing)
3. [Feature Engineering](#feature-engineering)
4. [Model Training](#model-training)
   - [Regression Model for Duration of Stay](#regression-model-for-duration-of-stay)
   - [Classification Model for Risk Levels](#classification-model-for-risk-levels)
5. [Model Evaluation](#model-evaluation)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Saving Models](#saving-models)
8. [Usage](#usage)
9. [Contributing](#contributing)
10. [License](#license)

## Installation

To get started with this project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/your_username/hospital-stay-prediction.git
cd hospital-stay-prediction
pip install -r requirements.txt
```

## Data Preprocessing

The dataset undergoes several preprocessing steps before being used for modeling:

1. **Loading Data:** The dataset is loaded from a CSV file.
   ```python
   df = pd.read_csv('modified_output.csv')
   ```

2. **Dropping Unnecessary Columns:** Columns like 'Serial Number', 'Admission Number', and dates are dropped as they don't contribute to the model.
   ```python
   df = df.drop(['Serial Number', 'Admission Number', 'Date of Admission', 'Date of Discharge', 'month year'], axis=1)
   ```

3. **Encoding Categorical Variables:** Categorical variables are one-hot encoded.
   ```python
   one_hot_df = pd.DataFrame(one_hot.fit_transform(categorical_df).toarray(), columns=one_hot.get_feature_names_out(categorical_features))
   ```

4. **Handling Missing Data:** Rows with missing data are removed.
   ```python
   transformed_df = transformed_df.dropna()
   ```

5. **Renaming and Replacing Columns:** Columns are renamed for easier access, and spaces are replaced with underscores.
   ```python
   transformed_df.columns = transformed_df.columns.str.replace(' ', '_').str.lower()
   ```

6. **Final Cleanup:** Unnecessary columns after encoding are dropped, and final adjustments are made.
   ```python
   transformed_df.drop(['gender_f', 'rural(r)_/urban(u)_r', 'type_of_admission-emergency/opd_o', 'outcome_discharge'], axis=1, inplace=True)
   ```

## Feature Engineering

Feature engineering involves creating new features from existing ones. For example, combining age with the duration of intensive unit stay:

```python
X_train['age_intensive_stay'] = X_train['age'] * X_train['duration_of_intensive_unit_stay']
```

## Model Training

### Regression Model for Duration of Stay

A Random Forest Regressor is used to predict the duration of hospital stays.

```python
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train, y_duration_train)
```

### Classification Model for Risk Levels

A Random Forest Classifier is used to categorize patients' risk levels.

```python
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_risk, y_risk_train)
```

## Model Evaluation

The models are evaluated using metrics such as Mean Absolute Error (MAE) for regression and accuracy for classification.

```python
from sklearn.metrics import mean_absolute_error, accuracy_score

mae = mean_absolute_error(y_duration_val, y_pred_val)
accuracy = accuracy_score(y_risk_val, y_pred_risk_val)
```

## Hyperparameter Tuning

Hyperparameters of the models are tuned using `RandomizedSearchCV` and `GridSearchCV` for optimal performance.

```python
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(estimator=regressor, param_distributions=param_grid, n_iter=100, cv=4, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_duration_train)
```

## Saving Models

Trained models are saved using the `joblib` library for later use.

```python
import joblib

joblib.dump(regressor, 'random_forest_regressor.joblib')
joblib.dump(classifier, 'risk_level_classification_model.joblib')
```

## Usage

To use the trained models, load them with `joblib` and make predictions on new data.

```python
regressor = joblib.load('random_forest_regressor.joblib')
predictions = regressor.predict(new_data)
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

This README provides an overview of the project's structure, including steps for data preprocessing, feature engineering, model training, and evaluation. It also includes instructions for installing dependencies and using the trained models.
