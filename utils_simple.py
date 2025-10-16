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
    
def search_pubmed(keyword,max_result=5):
    if not keyword:
        return []
    
    query = 'AND'.join(keyword)

    try:
        handle = Entrez.esearch(db="pubmed",term=query,retmax=max_result)
        result = Entrez.read(handle)

        if not result["IdList"]:
            return []
        
        fetch_handle  = Entrez.efetch(db="pubmed",id=result["IdList"],ret)