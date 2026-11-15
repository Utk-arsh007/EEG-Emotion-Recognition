"""
FastAPI Backend Application for EEG Emotion Recognition System
Serves prediction endpoints and mounts the frontend demonstration web interface.
"""

import os
import sys
import io
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

from backend.schemas import (
    HealthResponse,
    PredictionResponse,
    SampleTrialRequest,
    SamplesListResponse,
    SampleItem
)
from src.config import CONFIG
from src.predict import EEGPredictor


app = FastAPI(
    title="EEG Emotion Recognition API",
    description="Real-time multi-output affective dimension prediction from 32-channel EEG signals using Hybrid CNN-BiLSTM-Attention architecture.",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor globally
predictor = None

@app.on_event("startup")
def startup_event():
    global predictor
    ckpt_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "best_hybrid_torch.pt")
    scaler_path = os.path.join(PROJECT_ROOT, "data", "processed", "eeg_scaler.npz")
    predictor = EEGPredictor(
        checkpoint_path=ckpt_path,
        scaler_path=scaler_path,
        backend="torch"
    )
    print(f"[API Startup] Predictor loaded successfully.")


# -------------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=(predictor is not None and predictor.model is not None),
        model_architecture="Hybrid CNN-BiLSTM-Attention (32 EEG Channels)",
        eeg_channels=CONFIG.dataset.eeg_channels_count,
        sampling_rate_hz=CONFIG.dataset.sampling_rate,
        target_dimensions=CONFIG.dataset.target_names
    )


@app.get("/api/samples", response_model=SamplesListResponse)
def list_sample_trials():
    """Lists available preloaded sample DEAP .dat files for instant web demonstration."""
    sample_dir = PROJECT_ROOT / "sample_data"
    sample_files = sorted(list(sample_dir.glob("s*.dat")))
    
    samples_list = []
    for sf in sample_files:
        p_str = sf.stem.replace("s", "")
        samples_list.append(SampleItem(
            filename=sf.name,
            participant_id=f"Subject {p_str} (Demo)",
            num_trials=4,
            description=f"Synthetic DEAP-conforming 32-channel EEG recording ({sf.name})"
        ))

    return SamplesListResponse(samples=samples_list)


@app.post("/api/predict/sample", response_model=PredictionResponse)
def predict_sample(req: SampleTrialRequest):
    """Predicts emotional state for a chosen preloaded sample file and trial index."""
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor not initialized")

    sample_path = PROJECT_ROOT / "sample_data" / req.sample_filename
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample file {req.sample_filename} not found.")

    try:
        res = predictor.predict_from_file(str(sample_path), trial_idx=req.trial_index)
        return PredictionResponse(
            success=True,
            predictions=res["predictions"],
            uncertainty_std=res["uncertainty_std"],
            circumplex=res["circumplex"],
            windows_analyzed=res["windows_analyzed"],
            trajectory=res.get("trajectory"),
            disclaimer=res["disclaimer"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/predict/upload", response_model=PredictionResponse)
async def predict_uploaded_file(file: UploadFile = File(...)):
    """
    Accepts an uploaded EEG file (.dat or .npy).
    Validates format, checks for at least 32 channels, and runs the hybrid prediction pipeline.
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor not initialized")

    filename = file.filename.lower()
    contents = await file.read()

    try:
        if filename.endswith(".dat"):
            # Unpickle DEAP format
            data_dict = pickle.loads(contents, encoding="latin1")
            if not isinstance(data_dict, dict) or "data" not in data_dict:
                raise ValueError("Uploaded .dat file does not contain a dictionary with key 'data'.")
            raw_arr = np.asarray(data_dict["data"], dtype=np.float32)
            # If 3D (trials, channels, samples), take trial 0
            trial_eeg = raw_arr[0] if raw_arr.ndim == 3 else raw_arr

        elif filename.endswith(".npy"):
            raw_arr = np.load(io.BytesIO(contents))
            trial_eeg = raw_arr[0] if raw_arr.ndim == 3 else raw_arr

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file format. Please upload a DEAP preprocessed .dat or .npy file."
            )

        # Shape validation
        if trial_eeg.ndim != 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Expected 2D trial EEG signal (channels, time_samples), got {trial_eeg.shape}."
            )

        num_channels, num_samples = trial_eeg.shape
        if num_channels < 32:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input signal has {num_channels} channels. The trained DEAP model strictly requires 32 EEG channels."
            )

        if num_samples < 256:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Signal length is {num_samples} samples; minimum required is 256 samples (2.0s at 128 Hz)."
            )

        res = predictor.predict_trial(trial_eeg)

        return PredictionResponse(
            success=True,
            predictions=res["predictions"],
            uncertainty_std=res["uncertainty_std"],
            circumplex=res["circumplex"],
            windows_analyzed=res["windows_analyzed"],
            trajectory=res.get("trajectory"),
            disclaimer=res["disclaimer"]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/api/affect-guide")
def get_affect_guide():
    """Returns educational descriptions of the 4 continuous affective dimensions and Circumplex model."""
    return {
        "dimensions": {
            "Valence": {
                "range": "1.0 (Most Negative / Unpleasant) to 9.0 (Most Positive / Pleasant)",
                "description": "Reflects the intrinsic pleasantness or hedonic tone of the emotional state.",
                "neurophysiology": "Closely associated with Frontal Alpha Asymmetry (FAA). Left frontal activation indicates positive approach, while right frontal indicates negative withdrawal."
            },
            "Arousal": {
                "range": "1.0 (Deeply Calming / Inactive) to 9.0 (Highly Excited / Agitated)",
                "description": "Reflects the physiological and mental activation level or intensity of the emotion.",
                "neurophysiology": "Correlates with Beta wave (14-30 Hz) power and desynchronization in parietal and central cortices."
            },
            "Dominance": {
                "range": "1.0 (Submissive / Controlled / Helpless) to 9.0 (Empowered / In Control)",
                "description": "Represents the degree of subjective control or mastery over the emotional environment.",
                "neurophysiology": "Associated with frontal midline theta and power distribution in temporal lobe areas."
            },
            "Liking": {
                "range": "1.0 (Strongly Disliked) to 9.0 (Strongly Liked)",
                "description": "Subjective cognitive aesthetic appreciation and preference for the sensory stimulus.",
                "neurophysiology": "Modulated by higher cortical reward circuits and bilateral fronto-temporal coherence."
            }
        },
        "circumplex_quadrants": {
            "HVHA": "High Valence, High Arousal: Joy, Euphoria, Excitement, Enthusiasm",
            "LVHA": "Low Valence, High Arousal: Anger, Anxiety, Distress, Frustration",
            "LVLA": "Low Valence, Low Arousal: Sadness, Depression, Lethargy, Boredom",
            "HVLA": "High Valence, Low Arousal: Calmness, Serenity, Peace, Contentment"
        },
        "ethics_and_disclaimer": "This system is developed strictly for research, academic evaluation, and affective computing demonstration. It is not a clinical medical diagnostic device."
    }


# Mount Frontend UI Static Files
frontend_dir = PROJECT_ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
