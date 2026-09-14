from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware


from ocr_engine import parse_lab_report
from cnn_engine import process_xray

app = FastAPI(
    title="Swasthya Vision Microservice",
    description="Memory-Optimized Medical Vision Triage API",
    version="1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (good for hackathon testing)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health_check():
    """Simple endpoint to verify the server is alive on Render."""
    return {"status": "Active", "message": "Swasthya Vision API is running"}



@app.post("/vision/lab-report")
async def handle_lab_report(file: UploadFile = File(...)):
    """
    Accepts PDFs and Images. 
    Uses PyTesseract (Local OCR) + GPT-4o-mini (Cloud Structuring).
    """
    try:
        
        file_bytes = await file.read()
        result = parse_lab_report(file_bytes, file.filename)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
            
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server processing error: {str(e)}")



@app.post("/vision/xray")
async def handle_xray(file: UploadFile = File(...)):
    """
    Accepts Images (JPEG/PNG).
    Uses MobileNetV2 ONNX Runtime (Ultra-lightweight Local CNN).
    """
    try:
        
        if file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="X-Rays must be image files (JPG/PNG), not PDFs.")
            
        file_bytes = await file.read()
        
        #CNN Engine
        result = process_xray(file_bytes)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
            
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server processing error: {str(e)}")