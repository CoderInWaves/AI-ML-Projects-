import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_excel('/home/happy/Desktop/Dementia/ML Model/oasis_cross-sectional-5708aa0a98d82080.xlsx')

df['dementia'] = (df['CDR'] > 0).astype(int)

features = ['Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
X = df[features]
y = df['dementia']

mask = X.notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42
)

cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f'5-Fold CV Accuracy Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {test_accuracy:.4f}')

joblib.dump(model, '/home/happy/Desktop/Dementia/Final_project/dementia_tabular_model.pkl')

print('MODEL SAVED SUCCESSFULLY')
