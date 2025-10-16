import io
import base64
import uuid
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image as RPImage,Table
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
import os
import requests
from io import BytesIO
import json
import openai



try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try:
    from Bio import Entrez
    Entrez.email = "your_email@example.com"
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

def process_file(uploaded_file):
    """Process differnet file types"""
    ext = uploaded_file.name.split('.')[-1].lower()

    if ext in ['jpg','jpeg','png']:
        
        image = Image.open(uploaded_file).convert('RGB')
        return {"type":"image","data":image,"array":np.array(image)}
    elif ext in ['dcm'] and PYDICOM_AVAILABLE:
        try:
            bytes_data = uploaded_file.getvalue()
            with io.BytesIO(bytes_data) as dcm_bytes:
                dicom = pydicom.dcmread(dcm_bytes)
                image_array = dicom.pixel_array

                #convert to 8-bit for display
                image_array = ((image_array-image_array.min())/
                               (image_array.max()-image_array.min())*255).astype(np.uint8)
                
                return{
                    "type":"dicom",
                    "data":Image.fromarray(image_array),
                    "array":image_array,
                    "metadata":{
                        "PatientID": getattr(dicom,"PatientID","Unkown"),
                        "StudyDate":getattr(dicom,"StudyDate","Unknown"),
                        "Modality":getattr(dicom,"Modality","Unknown")
                    }
                }
        except Exception as e:
            print(f"Error processing DICOM : {e}")
            return None
        
    elif ext in ['nii','nii.gz'] and NIBABEL_AVAILABLE:
        #NIfTI FILE (3D Scan)
        try:
            bytes_data = uploaded_file.get_value()
            with io.BytesIO(bytes_data) as nii_bytes:
                #Save temp
                temp_path = f"temp_{uuid.uuid4()}.nii.gz"
                with open(temp_path,'wb') as f:
                    f.write(nii_bytes.read())
                
                #Load the NIfTI file
                nii_img = nib.load(temp_path)
                nii_data = nii_img.get_fdata()

                #Take a middle slice preview
                middle_slice = nii_data.shape[2] //2 
                image_array = nii_data[:,:,middle_slice]

                #Normalize for display
                image_array = ((image_array-image_array.min())/
                               (image_array.max()-image_array.min())*255).astype(np.uint8)
                
                #Clean temp
                os.remove(temp_path)

                return{
                    "type":"nifti",
                    "data":Image.fromarray(image_array),
                    "array": image_array,
                    "metadata":{
                        "Dimensions": nii_data.shape,
                        "Voxel Size": nii_img.header.get_zooms()
                    }
                }
        except Exception as e:
            print(f"Error processing NIfTI: {e}")
            return None
        
    elif ext in ['dcm','nii','nii.gz']:
        return{
            "type":"image",
            "data": Image.new('RGB',(400,400),color='gray'),
            "array":np.zeros((400,400,3),dtype=np.uint8),
            "metadata":{
                "Warning":"required libraries not installed for this file type",
                "Missing":"Install pydicom or nibabel to process this file"
            }
        }
    else:
        return None
    

def generate_heatmap(image_array):
    if len(image_array.shape) ==3:
        gray_image = cv2.cvtColor(image_array,cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image_array

    heatmap = cv2.applyColorMap(gray_image,cv2.COLORMAP_JET)

    if len(image_array.shape) == 2:
        image_array = cv2.cvtColor(image_array,cv2.COLOR_GRAY2RGB)

        overlay = cv2.addWeighted(heatmap,0.5,image_array,0.5,0)
    
    return Image.fromarray(overlay),Image.fromarray(heatmap)


def extract_findings_and_keywords(analysis_text):
    findings = []
    keywords = []

    # --- Extract Findings ---
    if "Impression" in analysis_text:
        # Split after "Impression:"
        impression_section = analysis_text.split("Impression")[1].strip()
        numbered_items = impression_section.split("\n")

        for item in numbered_items:
            item = item.strip()
            if item and (item[0].isdigit() or item[0] in ['-', '*']):
                # Remove number or bullet
                clean_item = item.lstrip("0123456789.-* )").strip()
                if clean_item:
                    findings.append(clean_item)

    # --- Keyword extraction ---
    base_keywords = [
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

    # Check which ones are present in the text
    for term in base_keywords:
        if term.lower() in analysis_text.lower():
            keywords.append(term)

    keywords = list(dict.fromkeys(keywords))

    return findings, keywords[:5]

def analyze_image(image,api_key,enable_xai=True):

    buffered = io.BytesIO()
    image.save(buffered,format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode()

    client = openai.OpenAI(api_key=api_key)

    prompt = """
    You are an expert radiologist. Analyze the provided medical image carefully.

    Provide a detailed structured report including:
    1. **Radiological Findings:** Describe all visible abnormalities.
    2. **Impression:** Summarize the key findings (e.g., pneumonia, cardiomegaly, fractures, tumors).
    3. **Possible Diagnoses:** Suggest possible conditions based on findings.
    4. **Recommendations:** Suggest follow-up actions (e.g., CT scan, clinical correlation).

    Format your respose with "Radiological Analysis " and "Impression" sections.
    """

    try:
        respose = client.chat.completions.create(
            model="chatgpt-4o-turbo",
            messages=[{
                "role":"user",
                "content":[
                    {
                        "type":"text",
                        "text":prompt
                     },
                     {
                         "type":"image_url",
                         "image_url":{"url":f"data:image/png;base64,{encoded_image}"}
                     }
                ]
            }],
            max_tokens=800,
        )

        analysis = respose.choices[0].message.content

        findings,keywords = extract_findings_and_keywords(analysis)

        return{
            "id":str(uuid.uuid4()),
            "analysis":analysis,
            "finding":findings,
            "keywords":keywords,
            "date":datetime.now().isoformat()
        }
    
    except Exception as e:
        return{
            "id":str(uuid.uuid4()),
            "analysis":f"Error analyzing image: {str(e)}",
            "finding":[],
            "keywords":[],
            "date":datetime.now().isoformat()
        }
    
def search_pubmed(keyword, max_result=5):
    if not keyword:
        return []

    query = ' AND '.join(keyword)

    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_result)
        result = Entrez.read(handle)

        if not result["IdList"]:
            return []

        fetch_handle = Entrez.efetch(db="pubmed", id=result["IdList"], rettype="medline", retmode="text")
        records = fetch_handle.read().split('\n\n')

        publications = []
        for record in records:
            if not record.strip():
                continue

            pub_data = {}
            lines = record.split('\n')
            for line in lines:
                if line.startswith("TI  -"):
                    pub_data["title"] = line.replace("TI  -", "").strip()
                elif line.startswith("AB  -"):
                    pub_data["abstract"] = line.replace("AB  -", "").strip()
                elif line.startswith("AU  -"):
                    pub_data.setdefault("authors", []).append(line.replace("AU  -", "").strip())
                elif line.startswith("DP  -"):
                    pub_data["year"] = line.replace("DP  -", "").strip()

            if pub_data:
                publications.append(pub_data)

        return publications

    except Exception as e:
        print(f"Error searching PubMed: {e}")
        return []

def search_clinical_trials(keywords,max_results=3):
    if not keywords:
        return []
    
    return[{
        "id":f"NCT{100+idx}",
        "title":f"Clinical Trial on {' '.join(keywords[:2])}",
        "status":"Recruting",
        "phase":f"Phase{idx+1}"}
        for idx in range(max_results)
    ]

def generate_report(data, include_references=True):
    """Generate a structured medical AI report as a PDF file"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=12,
        textColor=colors.darkblue
    )

    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=8,
        textColor=colors.darkred
    )

    normal = styles['Normal']
    story = []

    # ---- Header ----
    story.append(Paragraph("AI Radiology Report", title_style))
    story.append(Spacer(1, 12))

    # ---- Analysis Section ----
    story.append(Paragraph("<b>1️⃣ Radiological Analysis</b>", subtitle_style))
    story.append(Paragraph(data.get("analysis", "No analysis available"), normal))
    story.append(Spacer(1, 10))

    # ---- Findings ----
    if data.get("finding"):
        story.append(Paragraph("<b>2️⃣ Key Findings</b>", subtitle_style))
        findings_list = "<br/>".join([f"• {f}" for f in data["finding"]])
        story.append(Paragraph(findings_list, normal))
        story.append(Spacer(1, 10))

    # ---- Keywords ----
    if data.get("keywords"):
        story.append(Paragraph("<b>3️⃣ Keywords</b>", subtitle_style))
        story.append(Paragraph(", ".join(data["keywords"]), normal))
        story.append(Spacer(1, 10))

    # ---- Add Heatmap / Image ----
    if "heatmap" in data:
        story.append(Paragraph("<b>4️⃣ Visual Analysis (Heatmap)</b>", subtitle_style))
        img_buffer = io.BytesIO()
        data["heatmap"].save(img_buffer, format="PNG")
        img_buffer.seek(0)
        story.append(RPImage(img_buffer, width=400, height=300))
        story.append(Spacer(1, 10))

    # ---- PubMed Research ----
    if include_references and "pubmed" in data:
        story.append(Paragraph("<b>5️⃣ Related PubMed Studies</b>", subtitle_style))
        for pub in data["pubmed"]:
            title = pub.get("title", "Untitled")
            authors = ", ".join(pub.get("authors", []))
            year = pub.get("year", "N/A")
            abstract = pub.get("abstract", "")
            story.append(Paragraph(f"<b>{title}</b> ({year})<br/>{authors}<br/>{abstract}", normal))
            story.append(Spacer(1, 6))

    # ---- Clinical Trials ----
    if include_references and "trials" in data:
        story.append(Paragraph("<b>6️⃣ Related Clinical Trials</b>", subtitle_style))
        for trial in data["trials"]:
            story.append(Paragraph(
                f"🧪 <b>{trial['title']}</b> ({trial['id']})<br/>"
                f"Status: {trial['status']} | Phase: {trial['phase']}",
                normal
            ))
            story.append(Spacer(1, 6))

    # ---- Footer ----
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal))
    story.append(Paragraph("AI Medical Assistant © 2025", normal))

    doc.build(story)
    buffer.seek(0)
    return buffer

def get_analysis_store():
    if os.path.exists("analysis_store.json"):
        with open("analysis_store.json","r") as f:
            return json.load(f)
    return {"analyses":[]}

def save_analysis(analysis_data, filename="unknown.jpg"):
    """Save the AI analysis details to a local JSON file for record-keeping."""
    store = get_analysis_store()
    analysis_data["filename"] = filename
    store["analyses"].append(analysis_data)

    # Save to file
    with open("analysis_store.json", "w") as f:
        json.dump(store, f)
  
    return analysis_data

def get_analysis_by_id(analysis_id):
    store = get_analysis_store()

    for analysis in store["analyses"]:
        if analysis["id"] == analysis_id:
            return analysis
    
    return None

def get_latest_analyses(limit=5):
    """Fetch the latest N (default 5) analyses from the local store."""
    store = get_analysis_store()
    analyses = store.get("analyses", [])

    # Sort analyses by date (newest first)
    sorted_analyses = sorted(
        analyses,
        key=lambda x: x.get("date", ""),
        reverse=True
    )

    # Return top N
    return sorted_analyses[:limit]

from collections import Counter

def extract_common_findings(top_n=5):
    """Extract the most common findings across all saved analyses."""
    store = get_analysis_store()
    all_findings = []

    # Collect findings from all analyses
    for analysis in store.get("analyses", []):
        findings = analysis.get("findings", [])
        all_findings.extend(findings)

    if not all_findings:
        return []

    # Count the most frequent findings
    counter = Counter(all_findings)
    common_findings = counter.most_common(top_n)

    # Return as list of dicts for readability
    return [{"finding": f, "count": c} for f, c in common_findings]

def generate_static_report():
    """Generate a static summary report of all analyses (statistics + insights)."""
    store = get_analysis_store()
    analyses = store.get("analyses", [])
    
    # Prepare data
    total_cases = len(analyses)
    latest = get_latest_analyses(limit=5)
    common = extract_common_findings(top_n=5)

    # Create PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    normal = styles["Normal"]

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Heading1"],
        fontSize=20,
        spaceAfter=16,
        textColor=colors.darkblue
    )

    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Heading2"],
        fontSize=14,
        spaceAfter=8,
        textColor=colors.darkred
    )

    story = []

    # ---- Header ----
    story.append(Paragraph("📊 AI Radiology Summary Report", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal))
    story.append(Spacer(1, 12))

    # ---- Overview ----
    story.append(Paragraph("<b>1️⃣ Overview</b>", subtitle_style))
    story.append(Paragraph(f"Total Analyses Performed: <b>{total_cases}</b>", normal))
    story.append(Spacer(1, 8))

    # ---- Common Findings ----
    story.append(Paragraph("<b>2️⃣ Most Common Findings</b>", subtitle_style))
    if common:
        data = [["Finding", "Frequency"]] + [[f["finding"], str(f["count"])] for f in common]
        table = Table(data, hAlign="LEFT")
        table.setStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")
        ])
        story.append(table)
    else:
        story.append(Paragraph("No findings recorded yet.", normal))
    story.append(Spacer(1, 12))

    # ---- Latest Analyses ----
    story.append(Paragraph("<b>3️⃣ Recent Analyses</b>", subtitle_style))
    if latest:
        for a in latest:
            story.append(Paragraph(
                f"<b>Date:</b> {a.get('date', 'N/A')}<br/>"
                f"<b>File:</b> {a.get('filename', 'Unknown')}<br/>"
                f"<b>Keywords:</b> {', '.join(a.get('keywords', []))}<br/>"
                f"<b>Findings:</b> {', '.join(a.get('findings', []))[:120]}...",
                normal
            ))
            story.append(Spacer(1, 8))
    else:
        story.append(Paragraph("No recent analyses available.", normal))

    # ---- Footer ----
    story.append(Spacer(1, 20))
    story.append(Paragraph("AI Medical Assistant © 2025", normal))

    doc.build(story)
    buffer.seek(0)

    return buffer
