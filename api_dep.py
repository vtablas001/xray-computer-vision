import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image
from io import BytesIO

# Load the saved model
model_save_path = r'C:\Users\victor.tablas\OneDrive - United Nations Development Programme\Documents\cnn_pneumonia_model_VGG16.keras'

try:
    loaded_model = load_model(model_save_path)
except Exception as e:
    print(f"Error loading the model: {e}")
    loaded_model = None  # Fallback if model fails to load

# Initialize FastAPI app
app = FastAPI()

# Define response model
class PredictionResult(BaseModel):
    prediction: float
    class_label: str

@app.get("/")
async def read_root():
    return {"message": "Pneumonia Detection API"}

@app.post("/predict/", response_model=PredictionResult)
async def predict_image(file: UploadFile = File(...)):
    if loaded_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

        # Preprocess the image
        image = image.resize((224, 224))
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Predict
        prediction = loaded_model.predict(img_array)[0][0]
        class_label = "Pneumonia" if prediction > 0.5 else "Normal"

        return PredictionResult(prediction=float(prediction), class_label=class_label)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")



