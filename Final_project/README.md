# Dementia Prediction Web Application

## Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure models are in the current directory:
- dementia_tabular_model.pkl
- dementia_mri_model.pth

3. Run the application:
```bash
uvicorn app:app --reload
```

4. Open browser and navigate to:
```
http://127.0.0.1:8000
```

## Project Structure
```
Final_project/
├── app.py                          # FastAPI backend
├── requirements.txt                # Python dependencies
├── dementia_tabular_model.pkl      # Trained tabular model
├── dementia_mri_model.pth          # Trained MRI model
├── templates/
│   └── index.html                  # Frontend UI
├── static/
│   ├── style.css                   # Styling
│   └── script.js                   # Frontend logic
```

## Usage

1. Fill in clinical data (optional)
2. Upload MRI image (optional)
3. Select prediction mode
4. Click "Analyze Patient Data"
5. View results with confidence scores

## Notes

- At least one input method (clinical data OR MRI) is required
- Ensemble mode combines both models for best accuracy
- Results are for research purposes only
