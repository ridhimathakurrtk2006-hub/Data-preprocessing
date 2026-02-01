import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Add species column
df['species'] = iris.target_names[df['target']]

# Handle missing values (Iris has none, but for demo)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode species
encoder = LabelEncoder()
df['species'] = encoder.fit_transform(df['species'])

# Normalize features
scaler = MinMaxScaler()
df[df.columns[:-2]] = scaler.fit_transform(df[df.columns[:-2]])

# Save cleaned dataset
df.to_csv("data/cleaned_iris.csv", index=False)

# Train-test split
X = df.drop(columns=['species', 'target'])
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)