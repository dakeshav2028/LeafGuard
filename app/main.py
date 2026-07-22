import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from app.model_utils import ModelHandler

# Paths to the model and classes text file generated during training
MODEL_PATH = "c:/MY_PROJECTS/Image_Classification_web_app/models/plant_disease_model.keras"
CLASSES_PATH = "c:/MY_PROJECTS/Image_Classification_web_app/evaluation/classes.txt"

# Initialize our model helper instance
model_handler = ModelHandler(model_path=MODEL_PATH, classes_path=CLASSES_PATH)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager.
    Runs on server startup and shutdown. Loads the model once
    so subsequent request latencies are minimized.
    """
    try:
        model_handler.load_model()
    except Exception as e:
        print("WARNING: Model could not be loaded at startup.")
        print(f"Error details: {e}")
        print("Please ensure the training script has been executed and outputs are saved.")
    yield
    # Cleanup logic can go here if needed on shutdown

# Create the FastAPI instance
app = FastAPI(
    title="LeafGuard Plant Disease Classifier API",
    description="Backend API for predicting plant leaf disease using transfer learning on MobileNetV2.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS (Cross-Origin Resource Sharing)
# Allows the React/HTML frontend to interact with the API when served from a different port/host.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict endpoint. Accepts an image file, passes it to the ModelHandler,
    and returns a JSON payload containing the predicted class, confidence,
    and confidence scores for alternative classes.
    """
    # Verify the uploaded file type is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file must be an image. Received content-type: {file.content_type}"
        )
        
    try:
        # Read file contents as binary bytes
        image_bytes = await file.read()
        
        # Run inference
        results = model_handler.predict(image_bytes)
        return results
    except Exception as e:
        # Fallback for unexpected failures (e.g. corrupt image files, shapes, etc.)
        raise HTTPException(
            status_code=500,
            detail=f"Error executing model inference: {str(e)}"
        )

# Mount the static frontend files
# This serves the user interface at the root URL (http://localhost:8000/)
FRONTEND_DIR = "c:/MY_PROJECTS/Image_Classification_web_app/frontend"
if os.path.exists(FRONTEND_DIR):
    # Route for serving the main UI index file
    @app.get("/")
    async def serve_index():
        index_file = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "LeafGuard API is running, but frontend/index.html was not found."}

    # Mount remaining assets (style.css, app.js, images, etc.)
    # This must be mounted last so it doesn't shadow /predict
    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="static")
else:
    print(f"WARNING: Frontend folder not found at {FRONTEND_DIR}. Frontend will not be served.")
