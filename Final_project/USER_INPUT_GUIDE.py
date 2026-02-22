"""
Dementia Prediction System - User Input Requirements
=====================================================

This document outlines the input requirements for using both models.

MODEL 1: TABULAR MODEL (dementia_tabular_model.pkl)
---------------------------------------------------
Input Type: Clinical/Demographic Data (Structured)

Required Fields (7 features):
1. Age (int): Patient age in years
   - Range: 40-100
   - Example: 68

2. Educ (float): Education level
   - Range: 1-5
   - 1 = Less than high school
   - 2 = High school
   - 3 = Some college
   - 4 = Bachelor's degree
   - 5 = Graduate degree
   - Example: 2.0

3. SES (float): Socioeconomic Status
   - Range: 1-5 (1=highest, 5=lowest)
   - Example: 3.0

4. MMSE (float): Mini-Mental State Examination score
   - Range: 0-30
   - Higher = better cognitive function
   - Example: 24.0

5. eTIV (int): Estimated Total Intracranial Volume (cm³)
   - Range: ~1000-2000
   - Example: 1344

6. nWBV (float): Normalized Whole Brain Volume
   - Range: 0.6-0.9
   - Example: 0.743

7. ASF (float): Atlas Scaling Factor
   - Range: 0.8-1.6
   - Example: 1.306

Input Format:
{
    "Age": 68,
    "Educ": 2.0,
    "SES": 3.0,
    "MMSE": 24.0,
    "eTIV": 1344,
    "nWBV": 0.743,
    "ASF": 1.306
}

Output:
- Prediction: 0 (Normal) or 1 (Dementia)
- Confidence scores for each class


MODEL 2: MRI MODEL (dementia_mri_model.pth)
------------------------------------------
Input Type: Brain MRI Image

Required Format:
- File type: PNG, JPG, JPEG, or GIF
- Content: Brain scan (axial/transverse view preferred)
- Preprocessing: Will be automatically resized to 224x224

Acceptable Sources:
- Brain MRI scan (T1-weighted MPRAGE preferred)
- Already processed brain images from FSL_SEG or similar
- GIF/PNG exports from DICOM viewers

Input Method:
1. File upload: /path/to/brain_scan.png
2. Or direct image array: numpy array of shape (H, W, 3)

Output:
- Prediction: 0 (Normal) or 1 (Dementia)
- Confidence scores for each class


COMBINED PREDICTION WORKFLOW
----------------------------

Option 1: TABULAR ONLY
- Use when you have clinical data but no MRI
- Fastest, requires only 7 numbers
- Accuracy: ~81%

Option 2: MRI ONLY
- Use when you have brain scan but no clinical data
- Requires image upload
- Accuracy: ~83%

Option 3: ENSEMBLE (RECOMMENDED)
- Use both models and combine predictions
- Most accurate approach
- Methods:
  a) Average probabilities from both models
  b) Weighted average (e.g., 60% MRI + 40% Tabular)
  c) Voting: Final prediction if both agree

Example Combined Prediction:
- Tabular: 70% Dementia
- MRI: 85% Dementia
- Average: 77.5% Dementia → Predict: Dementia


USER INTERFACE REQUIREMENTS
---------------------------

Form Fields Needed:
1. Patient Demographics Section:
   - Age (number input, 40-100)
   - Education Level (dropdown: 1-5)
   - Socioeconomic Status (dropdown: 1-5)

2. Cognitive Assessment Section:
   - MMSE Score (number input, 0-30)

3. Brain Measurements Section:
   - eTIV (number input, optional if no MRI report)
   - nWBV (number input, optional if no MRI report)
   - ASF (number input, optional if no MRI report)

4. MRI Upload Section:
   - File upload button (accept: .png, .jpg, .jpeg, .gif)
   - Or drag-and-drop zone
   - Image preview after upload

5. Prediction Options:
   - Radio buttons:
     [ ] Use Tabular Data Only
     [ ] Use MRI Only
     [ ] Use Both (Ensemble) ← Default

6. Submit Button:
   - "Analyze Patient Data"

Output Display:
- Risk Level: LOW / MEDIUM / HIGH
- Confidence Score: XX%
- Detailed breakdown:
  * Tabular Model: Prediction + Confidence
  * MRI Model: Prediction + Confidence
  * Combined Result
- Explanation of key factors
- Disclaimer: "For research purposes only"


VALIDATION RULES
-----------------
- All numeric fields: Check min/max ranges
- MMSE: Warn if < 24 (cognitive impairment indicator)
- Age: Must be ≥ 40
- MRI: Check file size < 10MB
- MRI: Verify image is grayscale or RGB
- At least ONE input method required (tabular OR MRI)


API ENDPOINT STRUCTURE (if building web service)
------------------------------------------------

POST /predict/tabular
Body: {
    "Age": 68,
    "Educ": 2.0,
    "SES": 3.0,
    "MMSE": 24.0,
    "eTIV": 1344,
    "nWBV": 0.743,
    "ASF": 1.306
}

POST /predict/mri
Body: multipart/form-data with image file

POST /predict/ensemble
Body: {
    "tabular_data": {...},
    "mri_file": file
}
"""
