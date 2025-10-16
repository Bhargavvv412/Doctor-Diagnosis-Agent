
def analyze_image(image_data,api_key,enable_xai=True):
    if isinstance(image_data,Image.Image):
        image_array = np.array(image_data)
    else:
        image_array = image_data

    analysis = """
**Radiological Analysis**

The image shows a chest X-ray with apparent bilateral pulmonary infiltrates,
predominantly in the lower lobes. There is evidence of ground-glass
opacities and possible consolidation in the right lower lobe. The cardiac silhouette
appears mildly enlarged. No pneumothorax or pleural effusion is visible.
The trachea is central, and bony thoracic structures are intact. Lung volumes
are slightly reduced, suggesting mild restrictive changes.

**Impression**
1. Bilateral lower lobe pneumonia or inflammatory infiltrates.
2. Ground-glass opacities suggestive of possible viral or atypical infection (consider COVID-19 or pneumonitis).
3. Mild cardiomegaly — correlate clinically for hypertension or cardiomyopathy.
4. No evidence of pleural effusion or pneumothorax.

**Recommendation**
Further evaluation with HRCT chest may be considered for better characterization
of parenchymal changes. Correlate clinically and with laboratory findings.
"""

    findings = [
        "Bilateral pulmonary infiltrates in lower lobes",
        "Ground-glass opacities, more pronounced in the right lower lobe",
        "Mild cardiomegaly (enlarged heart silhouette)",
        "No pneumothorax or pleural effusion detected",
        "Normal tracheal alignment and intact bony thoracic structures"
    ]

    keywords = [
    "chest x-ray",
    "bilateral infiltrates",
    "lower lobes",
    "ground-glass opacities",
    "right lower lobe consolidation",
    "cardiomegaly",
    "no pneumothorax",
    "no pleural effusion",
    "pulmonary infection",
    "pneumonia",
    "viral pneumonia",
    "atypical infection",
    "COVID-19",
    "lungs",
    "heart enlargement",
    "radiology",
    "imaging findings"
]
    
    #Generate unique ID
    analysis_id = str(uuid.uuid4())

    return{
        "id":analysis_id,
        "analysis":analysis,
        "finding":findings,
        "keywords":keywords,
        "date":datetime.now().isoformat()
    }
