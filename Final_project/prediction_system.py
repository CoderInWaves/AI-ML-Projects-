import joblib
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd

class DementiaPredictionSystem:
    def __init__(self, tabular_model_path, mri_model_path):
        """
        Initialize both models
        
        Args:
            tabular_model_path: Path to .pkl file
            mri_model_path: Path to .pth file
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tabular model
        self.tabular_model = joblib.load(tabular_model_path)
        
        # Load MRI model
        self.mri_model = models.densenet121(pretrained=False)
        num_features = self.mri_model.classifier.in_features
        self.mri_model.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
        self.mri_model.load_state_dict(torch.load(mri_model_path, map_location=self.device))
        self.mri_model = self.mri_model.to(self.device)
        self.mri_model.eval()
        
        # MRI preprocessing
        self.mri_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict_tabular(self, patient_data):
        """
        Predict using clinical/demographic data
        
        Args:
            patient_data: dict with keys: Age, Educ, SES, MMSE, eTIV, nWBV, ASF
        
        Returns:
            dict: {
                'prediction': 0 or 1,
                'confidence': float,
                'probabilities': [prob_normal, prob_dementia]
            }
        """
        features = ['Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
        
        # Validate input
        for feature in features:
            if feature not in patient_data:
                raise ValueError(f"Missing required field: {feature}")
        
        # Create DataFrame
        df = pd.DataFrame([patient_data])
        X = df[features]
        
        # Predict
        prediction = self.tabular_model.predict(X)[0]
        probabilities = self.tabular_model.predict_proba(X)[0]
        confidence = probabilities[prediction]
        
        return {
            'prediction': int(prediction),
            'prediction_label': 'Dementia' if prediction == 1 else 'Normal',
            'confidence': float(confidence),
            'probabilities': {
                'normal': float(probabilities[0]),
                'dementia': float(probabilities[1])
            }
        }
    
    def predict_mri(self, image_path_or_array):
        """
        Predict using brain MRI image
        
        Args:
            image_path_or_array: Path to image file or numpy array
        
        Returns:
            dict: {
                'prediction': 0 or 1,
                'confidence': float,
                'probabilities': [prob_normal, prob_dementia]
            }
        """
        # Load image
        if isinstance(image_path_or_array, str):
            image = Image.open(image_path_or_array).convert('RGB')
        elif isinstance(image_path_or_array, np.ndarray):
            image = Image.fromarray(image_path_or_array).convert('RGB')
        else:
            raise ValueError("Input must be file path or numpy array")
        
        # Preprocess
        image_tensor = self.mri_transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.mri_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            prediction = torch.argmax(probabilities).item()
            confidence = probabilities[prediction].item()
        
        return {
            'prediction': int(prediction),
            'prediction_label': 'Dementia' if prediction == 1 else 'Normal',
            'confidence': float(confidence),
            'probabilities': {
                'normal': float(probabilities[0]),
                'dementia': float(probabilities[1])
            }
        }
    
    def predict_ensemble(self, patient_data=None, image_path=None, weights=(0.5, 0.5)):
        """
        Combined prediction using both models
        
        Args:
            patient_data: dict for tabular model (optional)
            image_path: path to MRI image (optional)
            weights: tuple (tabular_weight, mri_weight), must sum to 1
        
        Returns:
            dict with combined prediction
        """
        results = {}
        
        # Get individual predictions
        if patient_data:
            results['tabular'] = self.predict_tabular(patient_data)
        
        if image_path:
            results['mri'] = self.predict_mri(image_path)
        
        # If only one model available
        if len(results) == 1:
            model_name = list(results.keys())[0]
            return {
                'method': f'{model_name}_only',
                'prediction': results[model_name]['prediction'],
                'prediction_label': results[model_name]['prediction_label'],
                'confidence': results[model_name]['confidence'],
                'details': results
            }
        
        # Ensemble prediction
        tab_weight, mri_weight = weights
        
        prob_normal = (
            results['tabular']['probabilities']['normal'] * tab_weight +
            results['mri']['probabilities']['normal'] * mri_weight
        )
        prob_dementia = (
            results['tabular']['probabilities']['dementia'] * tab_weight +
            results['mri']['probabilities']['dementia'] * mri_weight
        )
        
        prediction = 1 if prob_dementia > prob_normal else 0
        confidence = max(prob_normal, prob_dementia)
        
        return {
            'method': 'ensemble',
            'prediction': prediction,
            'prediction_label': 'Dementia' if prediction == 1 else 'Normal',
            'confidence': float(confidence),
            'probabilities': {
                'normal': float(prob_normal),
                'dementia': float(prob_dementia)
            },
            'details': results
        }


# Example usage
if __name__ == '__main__':
    # Initialize system
    system = DementiaPredictionSystem(
        tabular_model_path='/home/happy/Desktop/Dementia/Final_project/dementia_tabular_model.pkl',
        mri_model_path='/home/happy/Desktop/Dementia/Final_project/dementia_mri_model.pth'
    )
    
    # Example 1: Tabular prediction
    print("="*60)
    print("EXAMPLE 1: TABULAR PREDICTION")
    print("="*60)
    patient = {
        'Age': 68,
        'Educ': 2.0,
        'SES': 3.0,
        'MMSE': 24.0,
        'eTIV': 1344,
        'nWBV': 0.743,
        'ASF': 1.306
    }
    result = system.predict_tabular(patient)
    print(f"Prediction: {result['prediction_label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Probabilities: Normal={result['probabilities']['normal']:.2%}, Dementia={result['probabilities']['dementia']:.2%}")
    
    # Example 2: MRI prediction
    print("\n" + "="*60)
    print("EXAMPLE 2: MRI PREDICTION")
    print("="*60)
    import os
    dataset_dir = '/home/happy/Desktop/Dementia/Final_project/dataset_mri/val/normal'
    if os.path.exists(dataset_dir):
        sample_images = [f for f in os.listdir(dataset_dir) if f.endswith('.png')]
        if sample_images:
            sample_image = os.path.join(dataset_dir, sample_images[0])
            result = system.predict_mri(sample_image)
            print(f"Image: {sample_images[0]}")
            print(f"Prediction: {result['prediction_label']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Probabilities: Normal={result['probabilities']['normal']:.2%}, Dementia={result['probabilities']['dementia']:.2%}")
    
    # Example 3: Ensemble prediction
    print("\n" + "="*60)
    print("EXAMPLE 3: ENSEMBLE PREDICTION")
    print("="*60)
    if os.path.exists(dataset_dir) and sample_images:
        result = system.predict_ensemble(
            patient_data=patient,
            image_path=sample_image,
            weights=(0.4, 0.6)  # 40% tabular, 60% MRI
        )
        print(f"Method: {result['method']}")
        print(f"Final Prediction: {result['prediction_label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probabilities: Normal={result['probabilities']['normal']:.2%}, Dementia={result['probabilities']['dementia']:.2%}")
        print("\nIndividual model results:")
        print(f"  Tabular: {result['details']['tabular']['prediction_label']} ({result['details']['tabular']['confidence']:.2%})")
        print(f"  MRI: {result['details']['mri']['prediction_label']} ({result['details']['mri']['confidence']:.2%})")
