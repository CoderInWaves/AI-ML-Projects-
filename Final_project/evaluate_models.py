import os
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# PART A - TABULAR MODEL EVALUATION
print("="*60)
print("TABULAR MODEL EVALUATION")
print("="*60)

excel_path = '/home/happy/Desktop/Dementia/ML Model/oasis_cross-sectional-5708aa0a98d82080.xlsx'
df = pd.read_excel(excel_path)

df['dementia'] = (df['CDR'] > 0).astype(int)

features = ['Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
X = df[features]
y = df['dementia']

mask = X.notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tabular_model_path = '/home/happy/Desktop/Dementia/Final_project/dementia_tabular_model.pkl'
tabular_model = joblib.load(tabular_model_path)

y_train_pred = tabular_model.predict(X_train)
y_test_pred = tabular_model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\nTrain Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

print("\nConfusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=['Normal', 'Dementia']))

if train_acc - test_acc > 0.1:
    tabular_status = "OVERFIT"
elif train_acc < 0.7 and test_acc < 0.7:
    tabular_status = "UNDERFIT"
else:
    tabular_status = "GOOD FIT"

print(f"\nTabular Model Diagnosis: {tabular_status}")

# PART B - MRI MODEL EVALUATION
print("\n" + "="*60)
print("MRI MODEL EVALUATION")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mri_model_path = '/home/happy/Desktop/Dementia/Final_project/dementia_mri_model.pth'
mri_model = models.densenet121(pretrained=False)
num_features = mri_model.classifier.in_features
mri_model.classifier = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 2)
)
mri_model.load_state_dict(torch.load(mri_model_path, map_location=device))
mri_model = mri_model.to(device)
mri_model.eval()

dataset_dir = '/home/happy/Desktop/Dementia/Final_project/dataset_mri'

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'val'), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

def evaluate_mri(model, loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

mri_train_acc = evaluate_mri(mri_model, train_loader, device)
mri_val_acc = evaluate_mri(mri_model, val_loader, device)

print(f"\nTrain Accuracy: {mri_train_acc:.4f}")
print(f"Validation Accuracy: {mri_val_acc:.4f}")

if mri_train_acc - mri_val_acc > 0.1:
    mri_status = "OVERFIT"
elif mri_train_acc < 0.7 and mri_val_acc < 0.7:
    mri_status = "UNDERFIT"
else:
    mri_status = "GOOD FIT"

print(f"\nMRI Model Diagnosis: {mri_status}")

# FINAL OUTPUT
print("\n" + "="*60)
print("FINAL EVALUATION SUMMARY")
print("="*60)
print(f"TABULAR MODEL STATUS: {tabular_status}")
print(f"MRI MODEL STATUS: {mri_status}")
print("="*60)
