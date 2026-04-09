import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings and errors
os.environ['PATH'] = '/usr/bin:' + os.environ.get('PATH', '')  # Ensure ffmpeg is in PATH
os.environ['FFMPEG_BINARY'] = '/usr/bin/ffmpeg'  # Set ffmpeg binary path

import ffmpeg

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import tempfile
import requests
from inference.transnetv2 import TransNetV2

app = FastAPI(title="TransNetV2 API", description="Video Scene Segmentation using TransNetV2")

class PredictRequest(BaseModel):
    url: str

# Initialize the model
model = TransNetV2()

@app.post("/predict")
async def predict_video(request: PredictRequest):
    """
    Provide a video URL and get scene segmentation predictions.
    """
    url = request.url
    # Download video from URL to temporary location
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_path = temp_file.name
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to download video: {str(e)}"}, status_code=400)

    try:
        # Run prediction
        video, single_frame_pred, all_frames_pred = model.predict_video(temp_path)

        # Convert predictions to scenes
        scenes = TransNetV2.predictions_to_scenes(single_frame_pred)

        # Prepare response
        result = {
            "video_frames": len(video),
            "scenes": scenes.tolist(),
            "single_frame_predictions": single_frame_pred.tolist(),
            "all_frames_predictions": all_frames_pred.tolist()
        }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        # Clean up temporary file
        os.unlink(temp_path)

@app.get("/")
def read_root():
    return {"message": "TransNetV2 API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)