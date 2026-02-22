from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List
import joblib
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import io
import os
from datetime import datetime

app = FastAPI(title="NeuroScan AI - Dementia Prediction System")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tabular_model = joblib.load('dementia_tabular_model.pkl')

mri_model = models.densenet121(pretrained=False)
num_features = mri_model.classifier.in_features
mri_model.classifier = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 2)
)
mri_model.load_state_dict(torch.load('dementia_mri_model.pth', map_location=device))
mri_model = mri_model.to(device)
mri_model.eval()

mri_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(
    patient_name: str = Form(...),
    age: float = Form(...),
    educ: float = Form(...),
    ses: float = Form(...),
    mmse: float = Form(...),
    etiv: float = Form(...),
    nwbv: float = Form(...),
    asf: float = Form(...),
    mri_files: List[UploadFile] = File(...)
):
    try:
        results = {}

        # ── Tabular model ──────────────────────────────────────────
        patient_data = pd.DataFrame([{
            'Age': age, 'Educ': educ, 'SES': ses,
            'MMSE': mmse, 'eTIV': etiv, 'nWBV': nwbv, 'ASF': asf
        }])
        X = patient_data[['Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']]

        tab_pred = tabular_model.predict(X)[0]
        tab_proba = tabular_model.predict_proba(X)[0]

        results['tabular'] = {
            'prediction': int(tab_pred),
            'label': 'Dementia' if tab_pred == 1 else 'Normal',
            'prob_normal': round(float(tab_proba[0]) * 100, 2),
            'prob_dementia': round(float(tab_proba[1]) * 100, 2),
        }

        # ── MRI model (average across all uploaded scans) ──────────
        mri_proba_list = []
        for mri_file in mri_files:
            contents = await mri_file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            tensor = mri_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                out = mri_model(tensor)
                prob = torch.softmax(out, dim=1)[0]
                mri_proba_list.append(prob)

        avg_norm  = float(sum(p[0] for p in mri_proba_list) / len(mri_proba_list))
        avg_dem   = float(sum(p[1] for p in mri_proba_list) / len(mri_proba_list))
        mri_pred  = 1 if avg_dem > avg_norm else 0

        results['mri'] = {
            'prediction': mri_pred,
            'label': 'Dementia' if mri_pred == 1 else 'Normal',
            'prob_normal': round(avg_norm * 100, 2),
            'prob_dementia': round(avg_dem * 100, 2),
            'num_scans': len(mri_files)
        }

        # ── Ensemble (60 % MRI + 40 % tabular) ────────────────────
        ens_norm = 0.4 * tab_proba[0] + 0.6 * avg_norm
        ens_dem  = 0.4 * tab_proba[1] + 0.6 * avg_dem
        ens_pred = 1 if ens_dem > ens_norm else 0

        results['ensemble'] = {
            'prediction': ens_pred,
            'label': 'Dementia' if ens_pred == 1 else 'Normal',
            'prob_normal': round(ens_norm * 100, 2),
            'prob_dementia': round(ens_dem * 100, 2),
        }

        # ── Consistency check ──────────────────────────────────────
        models_agree = (tab_pred == mri_pred)
        warning = None if models_agree else (
            "Models disagree. Recommendation: further clinical evaluation."
        )

        # ── Risk level (based on ensemble dementia probability) ────
        dem_pct = results['ensemble']['prob_dementia']
        if dem_pct < 35:
            risk_level = 'LOW'
        elif dem_pct < 65:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'HIGH'

        return JSONResponse(content={
            'success': True,
            'patient_name': patient_name,
            'patient_age': int(age),
            'mri_count': len(mri_files),
            'risk_level': risk_level,
            'final_prediction': results['ensemble']['label'],
            'final_prob_dementia': results['ensemble']['prob_dementia'],
            'final_prob_normal': results['ensemble']['prob_normal'],
            'warning': warning,
            'results': results,
            'timestamp': datetime.now().strftime('%B %d, %Y – %H:%M')
        })

    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e), "trace": traceback.format_exc()}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
