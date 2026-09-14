import onnxruntime as ort
import numpy as np
from PIL import Image
import io


ONNX_MODEL_PATH = "models/swasthya_vision_base.onnx"
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
CLASSES = ["COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

def process_xray(image_bytes: bytes):
    try:
        # 1. Open and resize image using Pillow (Extremely low RAM footprint)
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        
        # 2. Convert to Numpy Array & Normalize (Matching Colab transforms)
        img_data = np.array(img).astype('float32') / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_data = (img_data - mean) / std
        
        # 3. Reshape array to PyTorch/ONNX format: Batch, Channels, Height, Width (1, 3, 224, 224)
        img_data = np.transpose(img_data, (2, 0, 1))
        img_data = np.expand_dims(img_data, axis=0)
        
        # 4. Run the Lightweight Inference
        outputs = ort_session.run(None, {'input': img_data})
        logits = outputs[0][0]
        
        # 5. Extract the highest confidence prediction
        predicted_idx = np.argmax(logits)
        predicted_disease = CLASSES[predicted_idx]
        
        # Calculate a pseudo-confidence percentage
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()
        confidence = float(probabilities[predicted_idx] * 100)
        
        return {
            "success": True,
            "diagnosis": predicted_disease,
            "confidence": f"{confidence:.2f}%",
            "urgency_zone": "RED" if predicted_disease != "NORMAL" else "GREEN"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}