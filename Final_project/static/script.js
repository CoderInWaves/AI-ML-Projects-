const form = document.getElementById('predictionForm');
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('mriFile');
const imagePreview = document.getElementById('imagePreview');
const previewGrid = document.getElementById('previewGrid');
const fileCount = document.getElementById('fileCount');
const resultsCard = document.getElementById('resultsCard');
const errorCard = document.getElementById('errorCard');
const btnText = document.getElementById('btnText');
const btnLoader = document.getElementById('btnLoader');

let globalResultsData = null;

uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        previewImages(files);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        previewImages(e.target.files);
    }
});

function previewImages(files) {
    previewGrid.innerHTML = '';
    let validFiles = 0;
    
    Array.from(files).forEach((file, index) => {
        if (file.size > 10 * 1024 * 1024) {
            alert(`File ${file.name} is larger than 10MB and will be skipped`);
            return;
        }
        
        validFiles++;
        const reader = new FileReader();
        reader.onload = (e) => {
            const imgWrapper = document.createElement('div');
            imgWrapper.style.cssText = 'position: relative; border: 2px solid #e9ecef; border-radius: 8px; overflow: hidden;';
            
            const img = document.createElement('img');
            img.src = e.target.result;
            img.style.cssText = 'width: 100%; height: 150px; object-fit: cover;';
            
            const label = document.createElement('div');
            label.textContent = `Scan ${index + 1}`;
            label.style.cssText = 'position: absolute; bottom: 0; left: 0; right: 0; background: rgba(102, 126, 234, 0.9); color: white; padding: 5px; text-align: center; font-size: 0.9em; font-weight: bold;';
            
            imgWrapper.appendChild(img);
            imgWrapper.appendChild(label);
            previewGrid.appendChild(imgWrapper);
        };
        reader.readAsDataURL(file);
    });
    
    imagePreview.style.display = 'block';
    fileCount.textContent = `${validFiles} MRI scan(s) selected`;
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData(form);
    
    const hasTabular = formData.get('age') && formData.get('educ') && 
                       formData.get('ses') && formData.get('mmse') && 
                       formData.get('etiv') && formData.get('nwbv') && 
                       formData.get('asf');
    
    const hasMri = fileInput.files.length > 0;
    
    if (!hasTabular && !hasMri) {
        showError('Please provide either clinical data or MRI image');
        return;
    }
    
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Prediction failed');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
});

function displayResults(data) {
    globalResultsData = data;
    errorCard.style.display = 'none';
    
    // Display patient information
    document.getElementById('resultPatientName').textContent = data.patient_name;
    document.getElementById('resultPatientAge').textContent = data.patient_age;
    document.getElementById('resultDate').textContent = new Date().toLocaleDateString('en-US', { 
        year: 'numeric', month: 'long', day: 'numeric' 
    });
    document.getElementById('resultMriCount').textContent = data.mri_count + ' scan(s)';
    
    const riskIndicator = document.getElementById('riskIndicator');
    riskIndicator.className = 'risk-indicator ' + data.risk_level.toLowerCase();
    
    document.getElementById('riskLevel').textContent = data.risk_level;
    document.getElementById('riskLabel').textContent = data.final_prediction;
    document.getElementById('confidence').textContent = data.final_confidence.toFixed(1) + '%';
    
    // Populate tabular model results
    if (data.results.tabular) {
        const tabular = data.results.tabular;
        document.getElementById('tabularLabel').textContent = tabular.label;
        document.getElementById('tabularConf').textContent = (tabular.confidence * 100).toFixed(1) + '%';
        document.getElementById('tabularNormal').textContent = (tabular.prob_normal * 100).toFixed(1) + '%';
        document.getElementById('tabularDementia').textContent = (tabular.prob_dementia * 100).toFixed(1) + '%';
        
        // Animate progress bar
        setTimeout(() => {
            document.getElementById('tabularProgress').style.width = (tabular.confidence * 100) + '%';
            document.getElementById('tabularProgress').className = 'progress-fill ' + 
                (tabular.prediction === 1 ? 'high-risk' : 'low-risk');
        }, 300);
    }
    
    // Populate MRI model results
    if (data.results.mri) {
        const mri = data.results.mri;
        document.getElementById('mriLabel').textContent = mri.label;
        document.getElementById('mriConf').textContent = (mri.confidence * 100).toFixed(1) + '%';
        document.getElementById('mriNormal').textContent = (mri.prob_normal * 100).toFixed(1) + '%';
        document.getElementById('mriDementia').textContent = (mri.prob_dementia * 100).toFixed(1) + '%';
        
        // Animate progress bar
        setTimeout(() => {
            document.getElementById('mriProgress').style.width = (mri.confidence * 100) + '%';
            document.getElementById('mriProgress').className = 'progress-fill ' + 
                (mri.prediction === 1 ? 'high-risk' : 'low-risk');
        }, 500);
    }
    
    // Populate ensemble results
    if (data.results.ensemble) {
        const ensemble = data.results.ensemble;
        document.getElementById('ensembleLabel').textContent = ensemble.label;
        document.getElementById('ensembleConf').textContent = (ensemble.confidence * 100).toFixed(1) + '%';
        document.getElementById('ensembleNormal').textContent = (ensemble.prob_normal * 100).toFixed(1) + '%';
        document.getElementById('ensembleDementia').textContent = (ensemble.prob_dementia * 100).toFixed(1) + '%';
        
        // Animate progress bar
        setTimeout(() => {
            document.getElementById('ensembleProgress').style.width = (ensemble.confidence * 100) + '%';
            document.getElementById('ensembleProgress').className = 'progress-fill ' + 
                (ensemble.prediction === 1 ? 'high-risk' : 'low-risk');
        }, 700);
    }
    
    resultsCard.style.display = 'block';
    resultsCard.classList.add('fade-in');
    resultsCard.scrollIntoView({ behavior: 'smooth' });
}

function showError(message) {
    resultsCard.style.display = 'none';
    document.getElementById('errorMessage').textContent = message;
    errorCard.style.display = 'block';
    errorCard.scrollIntoView({ behavior: 'smooth' });
}

function hideError() {
    errorCard.style.display = 'none';
}

function resetForm() {
    form.reset();
    imagePreview.style.display = 'none';
    previewGrid.innerHTML = '';
    fileCount.textContent = '';
    resultsCard.style.display = 'none';
    errorCard.style.display = 'none';
    globalResultsData = null;
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function downloadReport() {
    if (!globalResultsData) {
        alert('No results available to download');
        return;
    }
    
    const data = globalResultsData;
    const date = new Date().toLocaleDateString('en-US');
    
    // Create report content
    let reportContent = `
DEMENTIA PREDICTION REPORT
${'='.repeat(80)}

PATIENT INFORMATION
${'-'.repeat(80)}
Patient Name: ${data.patient_name}
Age: ${data.patient_age} years
Date of Analysis: ${date}
Number of MRI Scans: ${data.mri_count}

PREDICTION RESULTS
${'-'.repeat(80)}
Final Prediction: ${data.final_prediction}
Overall Confidence: ${data.final_confidence.toFixed(2)}%
Risk Level: ${data.risk_level}

MODEL ANALYSIS
${'-'.repeat(80)}

1. RANDOMFOREST CLASSIFIER (Tabular Model)
   Prediction: ${data.results.tabular.label}
   Confidence: ${(data.results.tabular.confidence * 100).toFixed(2)}%
   Probabilities:
      - Normal: ${(data.results.tabular.prob_normal * 100).toFixed(2)}%
      - Dementia: ${(data.results.tabular.prob_dementia * 100).toFixed(2)}%

2. DENSENET-121 (Deep Learning Model)
   Prediction: ${data.results.mri.label}
   Confidence: ${(data.results.mri.confidence * 100).toFixed(2)}%
   Number of Scans Analyzed: ${data.results.mri.num_scans}
   Probabilities:
      - Normal: ${(data.results.mri.prob_normal * 100).toFixed(2)}%
      - Dementia: ${(data.results.mri.prob_dementia * 100).toFixed(2)}%

3. ENSEMBLE MODEL (Combined Prediction)
   Model Weights: 40% RandomForest + 60% DenseNet-121
   Prediction: ${data.results.ensemble.label}
   Confidence: ${(data.results.ensemble.confidence * 100).toFixed(2)}%
   Probabilities:
      - Normal: ${(data.results.ensemble.prob_normal * 100).toFixed(2)}%
      - Dementia: ${(data.results.ensemble.prob_dementia * 100).toFixed(2)}%

${'='.repeat(80)}
DISCLAIMER
${'-'.repeat(80)}
This report is generated for research purposes only and does not constitute 
a medical diagnosis. Please consult with a qualified healthcare professional 
for proper medical evaluation and diagnosis.

Generated by: Dementia Prediction System
Timestamp: ${new Date().toLocaleString('en-US')}
${'='.repeat(80)}
    `.trim();
    
    // Create and download text file
    const blob = new Blob([reportContent], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Dementia_Report_${data.patient_name.replace(/\s+/g, '_')}_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    
    // Show success message
    alert('✅ Report downloaded successfully!');
}
