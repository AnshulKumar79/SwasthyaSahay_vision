import os
import io
import pymupdf as fitz
from PIL import Image
import pytesseract
import json
import google.generativeai as genai
from pydantic import BaseModel
from typing import List, Literal

# ---------------- 1. DEFINE THE STRICT JSON STRUCTURE ----------------
# Pydantic schema forces GPT-4o-mini to return data EXACTLY in this format every single time.

class LabParameter(BaseModel):
    test_name: str
    observed_value: str
    reference_range: str
    status: Literal["NORMAL", "LOW", "HIGH", "CRITICAL"]
    clinical_interpretation: str

class TriageReport(BaseModel):
    document_type: str
    urgency_zone: Literal["GREEN", "YELLOW", "RED"]
    summary_text: str
    abnormalities: List[LabParameter]
    recommended_actions: List[str]

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------- 3. LOCAL TEXT EXTRACTION (Zero API Cost) ----------------
def extract_text_from_payload(file_bytes: bytes, filename: str) -> str:
    """Safely extracts text from PDFs and Images under 512MB RAM constraint."""
    extracted_text = ""
    try:
        if filename.lower().endswith(".pdf"):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            page = doc.load_page(0)  # Only read first page to save RAM
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            extracted_text = pytesseract.image_to_string(img.convert('L'))
            doc.close()

        elif filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(io.BytesIO(file_bytes))
            if img.width > 1600:
                ratio = 1600.0 / float(img.width)
                new_height = int((float(img.height) * float(ratio)))
                img = img.resize((1600, new_height), Image.Resampling.LANCZOS)
            extracted_text = pytesseract.image_to_string(img.convert('L'))
            
        return extracted_text
        
    except Exception as e:
        print(f"Extraction Error: {str(e)}")
        return ""

# ---------------- 4. CLOUD LLM PARSING ----------------
def parse_lab_report(file_bytes: bytes, filename: str) -> dict:
    """
    Sends the raw OCR text to GPT-4o-mini to extract parameters dynamically 
    based on the lab's own reference ranges.
    """
    # Step 1: Extract messy text locally
    raw_text = extract_text_from_payload(file_bytes, filename)
    
    if not raw_text.strip():
        return {"success": False, "error": "Could not extract readable text from the document."}

    try:
        # Initialize Gemini 1.5 Flash (Ultra-fast and lightweight)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        You are a highly accurate medical triage AI assisting ASHA workers in rural India.
        Read the following messy OCR text from a diagnostic report.
        1. Identify the document type (e.g., Complete Blood Count, Urine Routine, LFT, etc.).
        2. Extract every parameter, its value, and the reference range printed on the report.
        3. Flag any abnormalities ONLY if they fall outside the printed reference range.
        4. Determine the Urgency Zone (GREEN for normal, YELLOW for mild issues, RED for critical/severe issues).
        5. Provide actionable primary-care recommendations.
        
        Report Text:
        {raw_text}
        """
        
        # Call Gemini and enforce the exact Pydantic schema
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=TriageReport,
                temperature=0.1  # Low temperature for highly deterministic medical output
            )
        )
        
        # Gemini returns a JSON string, so we load it into a dictionary
        report_data = json.loads(response.text)
        report_data["success"] = True
        return report_data
        
    except Exception as e:
        print(f"LLM Parsing Error: {str(e)}")
        return {"success": False, "error": f"Failed to analyze report: {str(e)}"}