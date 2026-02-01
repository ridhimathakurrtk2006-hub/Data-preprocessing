# Iris Dataset Preprocessing Demo

This project demonstrates preprocessing steps on the classic Iris dataset.

## Steps

1. **Raw Dataset**
   - Contains 150 samples with 4 numeric features and 1 categorical target (`species`).

2. **Missing Value Handling**
   - Iris has no missing values, but pipeline includes `fillna` for robustness.

3. **Encoding**
   - `LabelEncoder` applied to `species` (setosa=0, versicolor=1, virginica=2).

4. **Normalization**
   - `MinMaxScaler` applied to all numeric features (scaled between 0–1).

5. **Train-Test Split**
   - Dataset split into training (80%) and testing (20%).

6. **Outputs**
   - `raw_iris.csv` → original dataset.
   - `cleaned_iris.csv` → processed dataset.
   - `preprocessing.py` → script with pipeline.
   - `iris_preprocessing_demo.ipynb` → notebook for interactive exploration.
