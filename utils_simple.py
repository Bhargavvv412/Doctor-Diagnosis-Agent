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
    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension in ['jpg','jpeg','png']:
        
        image = Image.open(uploaded_file)
        return {"type":"image","data":image,"array":np.array(image)}
    
    elif file_extension in ['dcm'] and PYDICOM_AVAILABLE:
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
        
    elif file_extension in ['nii','nii.gz'] and NIBABEL_AVAILABLE:
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
        
    elif file_extension in ['dcm','nii','nii.gz']:
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
    